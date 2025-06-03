import time
import grpc
from concurrent import futures
import Model_pb2 as pb2
import Service_pb2_grpc as pb2_grpc
import math


class InfoService(pb2_grpc.InfoServiceServicer):
    def FetchInfo(self, request, context):
        # Stream Info messages every 0.5 seconds
        while True:
            # Create a dummy Info object
            info = pb2.Info(
                attention=pb2.AttentionInfo(level=1.5 + math.sin(time.time()) * 1.5, minLevel=0, maxLevel=3),
                speaker=pb2.SpeakerInfo(isMuted=False, level=max(0.5 + math.sin(time.time()) * 0.5, 0)),
                temperature=pb2.TemperatureInfo(
                    car=35.5,
                    driver=36.7 + math.sin(time.time()) * 6,
                    passenger=36.2 - math.sin(time.time()) * 6,
                    carAverage=35.0,
                    driverAverage=36.5,
                    passengerAverage=36.0,
                ),
            )
            yield info  # Send the Info object
            time.sleep(0.5)  # Wait for 0.5 seconds


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pb2_grpc.add_InfoServiceServicer_to_server(InfoService(), server)
    server.add_insecure_port("127.0.0.1:6100")
    print("Server is running on port 6100...")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()

