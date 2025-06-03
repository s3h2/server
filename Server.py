#!/usr/bin/env python3

import time
import math
import asyncio
import logging
import cv2
import numpy as np
import queue
import threading
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.models import vit_b_16, ViT_B_16_Weights
from ultralytics import YOLO

from av import VideoFrame
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
    RTCConfiguration,
    RTCIceServer,
)

from grpclib.utils import graceful_exit
from grpclib.server import Stream, Status, Server

import Model_pb2 as pb2
import Service_grpc as pb2_grpc

import mediapipe as mp

ADDRESS = "127.0.0.1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

inference_status = {"result": 0, "confidence": 0.0, "landmarks": []}
inference_threshold = 0.9
inference_lock = threading.Lock()

mp_face = mp.solutions.face_mesh
FACE_MESH = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)

l_idxs = [33, 160, 158, 133, 153, 144]
r_idxs = [263, 387, 385, 362, 380, 373]

def compute_ear(lm, idxs):
    pts = np.array([(lm[i].x, lm[i].y) for i in idxs])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3]) + 1e-6
    return (A + B) / (2 * C)

def dummy_npwarn_decorator_factory():
    def npwarn_decorator(x):
        return x

    return npwarn_decorator


np._no_nep50_warning = getattr(np, "_no_nep50_warning", dummy_npwarn_decorator_factory)

pcs = set()


class InferenceVideoTrack(VideoStreamTrack):
    def __init__(
        self,
        capture_device=0,
        model=None,
        yolo_model=None,
        device=None,
        num_segments=3,
    ):
        super().__init__()
        self.cap = cv2.VideoCapture(capture_device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)

        self.frame_queue = queue.Queue(maxsize=10)
        self.display_queue = queue.Queue(maxsize=10)
        self.num_segments = num_segments
        self.model = model
        self.yolo_model = yolo_model
        self.device = device
        self.last_inference_label = ""

        threading.Thread(target=self._frame_capturer, daemon=True).start()
        threading.Thread(target=self._background_inference, daemon=True).start()

    def _detect_faces_yolo(self, frame):
        results = self.yolo_model.predict(frame)
        if len(results) == 0:
            return []
        detections = results[0].boxes.xyxy
        face_detections = []
        for row in detections:
            face_detections.append(row.long())
        return face_detections

    def _frame_capturer(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)
            if self.display_queue.full():
                try:
                    self.display_queue.get_nowait()
                except queue.Empty:
                    pass
            self.display_queue.put(frame)
            time.sleep(0.03)

    def _background_inference(self):
        segment_buffer = []
        while True:
            while len(segment_buffer) < self.num_segments:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    face_detections = self._detect_faces_yolo(frame)
                    if len(face_detections) > 0:
                        (x1, y1, x2, y2) = face_detections[0]
                        h, w = frame.shape[:2]
                        x1 = max(x1 - 20, 0)
                        y1 = max(y1 - 20, 0)
                        x2 = min(x2 + 20, w)
                        y2 = min(y2 + 20, h)
                        face = frame[y1:y2, x1:x2]
                        segment_buffer.append(face)
                else:
                    time.sleep(0.01)

            if len(segment_buffer) >= self.num_segments:
                buffer_copy = segment_buffer[:self.num_segments]
                segment_buffer.clear()

                image_tensors = []
                mesh_feats = []
                for face in buffer_copy:
                    tensor = torch.from_numpy(face).permute(2, 0, 1).float() / 255
                    tensor = TF.normalize(
                        tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    )
                    tensor = TF.resize(tensor, (224, 224))
                    image_tensors.append(tensor)
                    res = FACE_MESH.process(face)
                    feat = [0.0] * 4
                    if res.multi_face_landmarks:
                        lm = res.multi_face_landmarks[0].landmark
                        landmark_flat = []
                        for i in range(len(lm)):
                            landmark_flat.extend([lm[i].x, lm[i].y, lm[i].z])
                        inference_status["landmarks"] = landmark_flat
                        ear_l = compute_ear(lm, l_idxs)
                        ear_r = compute_ear(lm, r_idxs)
                        feat = [ear_l, ear_r, (ear_l+ear_r)/2, max(ear_l,ear_r)]
                        mesh_feats.append(torch.tensor(feat, device=device, dtype=torch.float32))
                
                if len(image_tensors) == self.num_segments:
                    with torch.no_grad():
                        image_tensors = torch.stack(image_tensors).unsqueeze(0).to(self.device)
                        mesh_tensors = torch.stack(mesh_feats).unsqueeze(0).to(self.device)
                        output, confidence = model(image_tensors, mesh_tensors)

                    _, predicted = torch.max(output, 1)
                    predicted_class = predicted.item()
                    inference_status["result"] = predicted_class if confidence.item() > inference_threshold else 1 - predicted_class
                    inference_status["confidence"] = confidence.item()
                    self.last_inference_label = (
                        f"STATE: {predicted_class} ({confidence.item():.2f})"
                    )
                    print(self.last_inference_label)
            else:
                time.sleep(0.1)

    async def recv(self):
        ann_frame = self.display_queue.get()
        label = self.last_inference_label
        if label:
            cv2.putText(ann_frame, label, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        frame = VideoFrame.from_ndarray(ann_frame, format="bgr24")
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        return frame


class InfoService(pb2_grpc.InfoServiceBase):
    async def FetchInfo(self, stream: Stream):
        print("FETCH_INFO")

        while True:
            current_status = inference_status
            status_result = current_status["result"]
            info = pb2.Info(
                attention=pb2.AttentionInfo(
                    level=status_result, minLevel=0, maxLevel=3
                ),
                speaker=pb2.SpeakerInfo(
                    isMuted=False, level=max(0.5 + math.sin(time.time()) * 0.5, 0)
                ),
                temperature=pb2.TemperatureInfo(
                    car=35.5,
                    driver=36.7,
                    passenger=36.2,
                    carAverage=35.0,
                    driverAverage=36.5,
                    passengerAverage=36.0,
                ),
                modelInfo=pb2.ModelInfo(
                    ear=0,
                    cls=current_status["result"],
                    cof=current_status["confidence"],
                    landmarks=current_status["landmarks"],
                ),
            )
            try:
                await stream.send_message(info)
                await asyncio.sleep(0.5)
            except():
                await stream.send_trailing_metadata(status=Status.CANCELLED)

    async def Offer(self, stream: Stream):
        print("OFFER")
        request = await stream.recv_message()
        print(request)
        offer = RTCSessionDescription(sdp=request.sdp, type=request.type)
        print("Receive a offer: ")

        config = RTCConfiguration(
            iceServers=[RTCIceServer("stun:stun.ciktel.com:3478")]
        )
        pc = RTCPeerConnection(config)
        pcs.add(pc)

        @pc.on("icecandidate")
        async def on_icecandidate(event):
            candidate = event.candidate
            if candidate:
                print("ICE CANDIDATE GENERATED:", candidate)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logging.info("Connection state is %s", pc.connectionState)

        await pc.setRemoteDescription(offer)
        print("Set a remote description from offer")

        pc.addTrack(
            InferenceVideoTrack(device=device, model=model, yolo_model=yolo_model)
        )
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        while pc.iceGatheringState != "complete":
            await asyncio.sleep(0.1)
        print("Create an answer and set local description")

        await stream.send_message(
            pb2.MirrorInfo(sdp=pc.localDescription.sdp, type=pc.localDescription.type)
        )

    async def UpdateInfo(self, stream: Stream):
        print("UpdateInfo")


class AttentionPooling(torch.nn.Module):
    def __init__(self, feature_dim, num_segments):
        super(AttentionPooling, self).__init__()
        self.attention_weights = torch.nn.Parameter(
            torch.ones(num_segments, feature_dim), requires_grad=True
        )

    def forward(self, x):
        attention_scores = F.softmax(self.attention_weights, dim=0)
        weighted_features = x * attention_scores.unsqueeze(0)
        aggregated_features = weighted_features.sum(dim=1)
        return aggregated_features


class TSN(torch.nn.Module):
    def __init__(self, num_segments=3, num_classes=2):
        super(TSN, self).__init__()
        self.num_segments = num_segments
        self.base_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        num_features = self.base_model.heads.head.in_features
        self.base_model.heads.head = torch.nn.Identity()
        self.attn_pool = AttentionPooling(num_features, num_segments)
        self.head = torch.nn.Linear(num_features, num_classes)

    def forward(self, x):
        batch_size, num_segments, C, H, W = x.shape
        x = x.view(batch_size * num_segments, C, H, W)
        features = self.base_model(x)
        features = features.view(batch_size, num_segments, -1)

        pooled = self.attn_pool(features)
        out = self.head(pooled)

        probabilities = F.softmax(out, dim=1)
        confidences, _ = torch.max(probabilities, dim=1)
        return out, confidences, pooled


class TSNWithMesh(torch.nn.Module):
    def __init__(self, num_segments=3, num_classes=2, mesh_dim=4):
        super().__init__()
        self.tsn = TSN(num_segments, num_classes)
        feat_dim = self.tsn.head.in_features
        self.classifier = torch.nn.Linear(feat_dim + mesh_dim, num_classes)

    def forward(self, x, mesh):
        out, conf, img_feats = self.tsn(x)
        combined = torch.cat([img_feats, mesh], dim=1)
        logits = self.classifier(combined)
        probs = F.softmax(logits, dim=1)
        return logits, probs.max(dim=1)[0]


def load_yolo_model(device):
    yolo_model = YOLO("face.pt")
    yolo_model.to(device)
    return yolo_model


async def run_grpc_server():
    server = Server([InfoService()])
    with graceful_exit([server]):
        await server.start(ADDRESS, 6100)
        print(f"Serving on {ADDRESS}:6100")
        await server.wait_closed()


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = torch.nn.DataParallel(TSNWithMesh().to(device))
    model.load_state_dict(torch.load("run.pth", map_location=device))
    model.eval()
    print(model)
    max_queue_size = 3
    yolo_model = load_yolo_model(device)
    asyncio.run(run_grpc_server())
