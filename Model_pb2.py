# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: Model.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0bModel.proto\"\x07\n\x05\x45mpty\":\n\x03\x42ox\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\r\n\x05width\x18\x03 \x01(\x02\x12\x0e\n\x06height\x18\x04 \x01(\x02\"\'\n\nMirrorInfo\x12\x0b\n\x03sdp\x18\x01 \x01(\t\x12\x0c\n\x04type\x18\x02 \x01(\t\"B\n\rAttentionInfo\x12\r\n\x05level\x18\x01 \x01(\x01\x12\x10\n\x08minLevel\x18\x02 \x01(\x07\x12\x10\n\x08maxLevel\x18\x03 \x01(\x07\"-\n\x0bSpeakerInfo\x12\x0f\n\x07isMuted\x18\x01 \x01(\x08\x12\r\n\x05level\x18\x02 \x01(\x02\"\x86\x01\n\x0fTemperatureInfo\x12\x0b\n\x03\x63\x61r\x18\x01 \x01(\x01\x12\x0e\n\x06\x64river\x18\x02 \x01(\x01\x12\x11\n\tpassenger\x18\x03 \x01(\x01\x12\x12\n\ncarAverage\x18\x04 \x01(\x01\x12\x15\n\rdriverAverage\x18\x05 \x01(\x01\x12\x18\n\x10passengerAverage\x18\x06 \x01(\x01\"E\n\tModelInfo\x12\x0b\n\x03\x65\x61r\x18\x01 \x01(\x01\x12\x0b\n\x03\x63ls\x18\x02 \x01(\x01\x12\x0b\n\x03\x63of\x18\x03 \x01(\x01\x12\x11\n\tlandmarks\x18\x04 \x03(\x02\"\x8e\x01\n\x04Info\x12!\n\tattention\x18\x01 \x01(\x0b\x32\x0e.AttentionInfo\x12\x1d\n\x07speaker\x18\x02 \x01(\x0b\x32\x0c.SpeakerInfo\x12%\n\x0btemperature\x18\x03 \x01(\x0b\x32\x10.TemperatureInfo\x12\x1d\n\tmodelInfo\x18\x04 \x01(\x0b\x32\n.ModelInfob\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'Model_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _EMPTY._serialized_start=15
  _EMPTY._serialized_end=22
  _BOX._serialized_start=24
  _BOX._serialized_end=82
  _MIRRORINFO._serialized_start=84
  _MIRRORINFO._serialized_end=123
  _ATTENTIONINFO._serialized_start=125
  _ATTENTIONINFO._serialized_end=191
  _SPEAKERINFO._serialized_start=193
  _SPEAKERINFO._serialized_end=238
  _TEMPERATUREINFO._serialized_start=241
  _TEMPERATUREINFO._serialized_end=375
  _MODELINFO._serialized_start=377
  _MODELINFO._serialized_end=446
  _INFO._serialized_start=449
  _INFO._serialized_end=591
# @@protoc_insertion_point(module_scope)
