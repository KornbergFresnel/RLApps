# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: manager.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
    name="manager.proto",
    package="",
    syntax="proto3",
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n\rmanager.proto\x1a\x1bgoogle/protobuf/empty.proto"#\n\x11PSRODistillString\x12\x0e\n\x06string\x18\x01 \x01(\t"C\n\x1dPSRODistillPlayerAndPolicyNum\x12\x0e\n\x06player\x18\x01 \x01(\x03\x12\x12\n\npolicy_num\x18\x02 \x01(\x03"#\n\x11PSRODistillPlayer\x12\x0e\n\x06player\x18\x01 \x01(\x03"5\n\x19PSRODistillPolicySpecJson\x12\x18\n\x10policy_spec_json\x18\x01 \x01(\t"Q\n\x19PSRODistillPolicySpecList\x12\x34\n\x10policy_spec_list\x18\x01 \x03(\x0b\x32\x1a.PSRODistillPolicySpecJson"\xb6\x01\n PSRODistillNewBestResponseParams\x12>\n\x1ametanash_specs_for_players\x18\x01 \x01(\x0b\x32\x1a.PSRODistillPolicySpecList\x12>\n\x1a\x64\x65legate_specs_for_players\x18\x02 \x03(\x0b\x32\x1a.PSRODistillPolicySpecList\x12\x12\n\npolicy_num\x18\x03 \x01(\x03"]\n PSRODistillPolicyMetadataRequest\x12\x0e\n\x06player\x18\x01 \x01(\x03\x12\x12\n\npolicy_num\x18\x02 \x01(\x03\x12\x15\n\rmetadata_json\x18\x03 \x01(\t")\n\x17PSRODistillConfirmation\x12\x0e\n\x06result\x18\x01 \x01(\x08",\n\x13PSRODistillMetadata\x12\x15\n\rjson_metadata\x18\x01 \x01(\t2\x92\x03\n\x12PSRODistillManager\x12\x39\n\tGetLogDir\x12\x16.google.protobuf.Empty\x1a\x12.PSRODistillString"\x00\x12\x44\n\x12GetManagerMetaData\x12\x16.google.protobuf.Empty\x1a\x14.PSRODistillMetadata"\x00\x12X\n\x1d\x43laimNewActivePolicyForPlayer\x12\x12.PSRODistillPlayer\x1a!.PSRODistillNewBestResponseParams"\x00\x12T\n\x13SubmitFinalBRPolicy\x12!.PSRODistillPolicyMetadataRequest\x1a\x18.PSRODistillConfirmation"\x00\x12K\n\rIsPolicyFixed\x12\x1e.PSRODistillPlayerAndPolicyNum\x1a\x18.PSRODistillConfirmation"\x00\x62\x06proto3',
    dependencies=[
        google_dot_protobuf_dot_empty__pb2.DESCRIPTOR,
    ],
)


_PSRODISTILLSTRING = _descriptor.Descriptor(
    name="PSRODistillString",
    full_name="PSRODistillString",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="string",
            full_name="PSRODistillString.string",
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=46,
    serialized_end=81,
)


_PSRODISTILLPLAYERANDPOLICYNUM = _descriptor.Descriptor(
    name="PSRODistillPlayerAndPolicyNum",
    full_name="PSRODistillPlayerAndPolicyNum",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="player",
            full_name="PSRODistillPlayerAndPolicyNum.player",
            index=0,
            number=1,
            type=3,
            cpp_type=2,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="policy_num",
            full_name="PSRODistillPlayerAndPolicyNum.policy_num",
            index=1,
            number=2,
            type=3,
            cpp_type=2,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=83,
    serialized_end=150,
)


_PSRODISTILLPLAYER = _descriptor.Descriptor(
    name="PSRODistillPlayer",
    full_name="PSRODistillPlayer",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="player",
            full_name="PSRODistillPlayer.player",
            index=0,
            number=1,
            type=3,
            cpp_type=2,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=152,
    serialized_end=187,
)


_PSRODISTILLPOLICYSPECJSON = _descriptor.Descriptor(
    name="PSRODistillPolicySpecJson",
    full_name="PSRODistillPolicySpecJson",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="policy_spec_json",
            full_name="PSRODistillPolicySpecJson.policy_spec_json",
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=189,
    serialized_end=242,
)


_PSRODISTILLPOLICYSPECLIST = _descriptor.Descriptor(
    name="PSRODistillPolicySpecList",
    full_name="PSRODistillPolicySpecList",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="policy_spec_list",
            full_name="PSRODistillPolicySpecList.policy_spec_list",
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=244,
    serialized_end=325,
)


_PSRODISTILLNEWBESTRESPONSEPARAMS = _descriptor.Descriptor(
    name="PSRODistillNewBestResponseParams",
    full_name="PSRODistillNewBestResponseParams",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="metanash_specs_for_players",
            full_name="PSRODistillNewBestResponseParams.metanash_specs_for_players",
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="delegate_specs_for_players",
            full_name="PSRODistillNewBestResponseParams.delegate_specs_for_players",
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="policy_num",
            full_name="PSRODistillNewBestResponseParams.policy_num",
            index=2,
            number=3,
            type=3,
            cpp_type=2,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=328,
    serialized_end=510,
)


_PSRODISTILLPOLICYMETADATAREQUEST = _descriptor.Descriptor(
    name="PSRODistillPolicyMetadataRequest",
    full_name="PSRODistillPolicyMetadataRequest",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="player",
            full_name="PSRODistillPolicyMetadataRequest.player",
            index=0,
            number=1,
            type=3,
            cpp_type=2,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="policy_num",
            full_name="PSRODistillPolicyMetadataRequest.policy_num",
            index=1,
            number=2,
            type=3,
            cpp_type=2,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="metadata_json",
            full_name="PSRODistillPolicyMetadataRequest.metadata_json",
            index=2,
            number=3,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=512,
    serialized_end=605,
)


_PSRODISTILLCONFIRMATION = _descriptor.Descriptor(
    name="PSRODistillConfirmation",
    full_name="PSRODistillConfirmation",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="result",
            full_name="PSRODistillConfirmation.result",
            index=0,
            number=1,
            type=8,
            cpp_type=7,
            label=1,
            has_default_value=False,
            default_value=False,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=607,
    serialized_end=648,
)


_PSRODISTILLMETADATA = _descriptor.Descriptor(
    name="PSRODistillMetadata",
    full_name="PSRODistillMetadata",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="json_metadata",
            full_name="PSRODistillMetadata.json_metadata",
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=650,
    serialized_end=694,
)

_PSRODISTILLPOLICYSPECLIST.fields_by_name[
    "policy_spec_list"
].message_type = _PSRODISTILLPOLICYSPECJSON
_PSRODISTILLNEWBESTRESPONSEPARAMS.fields_by_name[
    "metanash_specs_for_players"
].message_type = _PSRODISTILLPOLICYSPECLIST
_PSRODISTILLNEWBESTRESPONSEPARAMS.fields_by_name[
    "delegate_specs_for_players"
].message_type = _PSRODISTILLPOLICYSPECLIST
DESCRIPTOR.message_types_by_name["PSRODistillString"] = _PSRODISTILLSTRING
DESCRIPTOR.message_types_by_name[
    "PSRODistillPlayerAndPolicyNum"
] = _PSRODISTILLPLAYERANDPOLICYNUM
DESCRIPTOR.message_types_by_name["PSRODistillPlayer"] = _PSRODISTILLPLAYER
DESCRIPTOR.message_types_by_name[
    "PSRODistillPolicySpecJson"
] = _PSRODISTILLPOLICYSPECJSON
DESCRIPTOR.message_types_by_name[
    "PSRODistillPolicySpecList"
] = _PSRODISTILLPOLICYSPECLIST
DESCRIPTOR.message_types_by_name[
    "PSRODistillNewBestResponseParams"
] = _PSRODISTILLNEWBESTRESPONSEPARAMS
DESCRIPTOR.message_types_by_name[
    "PSRODistillPolicyMetadataRequest"
] = _PSRODISTILLPOLICYMETADATAREQUEST
DESCRIPTOR.message_types_by_name["PSRODistillConfirmation"] = _PSRODISTILLCONFIRMATION
DESCRIPTOR.message_types_by_name["PSRODistillMetadata"] = _PSRODISTILLMETADATA
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PSRODistillString = _reflection.GeneratedProtocolMessageType(
    "PSRODistillString",
    (_message.Message,),
    {
        "DESCRIPTOR": _PSRODISTILLSTRING,
        "__module__": "manager_pb2"
        # @@protoc_insertion_point(class_scope:PSRODistillString)
    },
)
_sym_db.RegisterMessage(PSRODistillString)

PSRODistillPlayerAndPolicyNum = _reflection.GeneratedProtocolMessageType(
    "PSRODistillPlayerAndPolicyNum",
    (_message.Message,),
    {
        "DESCRIPTOR": _PSRODISTILLPLAYERANDPOLICYNUM,
        "__module__": "manager_pb2"
        # @@protoc_insertion_point(class_scope:PSRODistillPlayerAndPolicyNum)
    },
)
_sym_db.RegisterMessage(PSRODistillPlayerAndPolicyNum)

PSRODistillPlayer = _reflection.GeneratedProtocolMessageType(
    "PSRODistillPlayer",
    (_message.Message,),
    {
        "DESCRIPTOR": _PSRODISTILLPLAYER,
        "__module__": "manager_pb2"
        # @@protoc_insertion_point(class_scope:PSRODistillPlayer)
    },
)
_sym_db.RegisterMessage(PSRODistillPlayer)

PSRODistillPolicySpecJson = _reflection.GeneratedProtocolMessageType(
    "PSRODistillPolicySpecJson",
    (_message.Message,),
    {
        "DESCRIPTOR": _PSRODISTILLPOLICYSPECJSON,
        "__module__": "manager_pb2"
        # @@protoc_insertion_point(class_scope:PSRODistillPolicySpecJson)
    },
)
_sym_db.RegisterMessage(PSRODistillPolicySpecJson)

PSRODistillPolicySpecList = _reflection.GeneratedProtocolMessageType(
    "PSRODistillPolicySpecList",
    (_message.Message,),
    {
        "DESCRIPTOR": _PSRODISTILLPOLICYSPECLIST,
        "__module__": "manager_pb2"
        # @@protoc_insertion_point(class_scope:PSRODistillPolicySpecList)
    },
)
_sym_db.RegisterMessage(PSRODistillPolicySpecList)

PSRODistillNewBestResponseParams = _reflection.GeneratedProtocolMessageType(
    "PSRODistillNewBestResponseParams",
    (_message.Message,),
    {
        "DESCRIPTOR": _PSRODISTILLNEWBESTRESPONSEPARAMS,
        "__module__": "manager_pb2"
        # @@protoc_insertion_point(class_scope:PSRODistillNewBestResponseParams)
    },
)
_sym_db.RegisterMessage(PSRODistillNewBestResponseParams)

PSRODistillPolicyMetadataRequest = _reflection.GeneratedProtocolMessageType(
    "PSRODistillPolicyMetadataRequest",
    (_message.Message,),
    {
        "DESCRIPTOR": _PSRODISTILLPOLICYMETADATAREQUEST,
        "__module__": "manager_pb2"
        # @@protoc_insertion_point(class_scope:PSRODistillPolicyMetadataRequest)
    },
)
_sym_db.RegisterMessage(PSRODistillPolicyMetadataRequest)

PSRODistillConfirmation = _reflection.GeneratedProtocolMessageType(
    "PSRODistillConfirmation",
    (_message.Message,),
    {
        "DESCRIPTOR": _PSRODISTILLCONFIRMATION,
        "__module__": "manager_pb2"
        # @@protoc_insertion_point(class_scope:PSRODistillConfirmation)
    },
)
_sym_db.RegisterMessage(PSRODistillConfirmation)

PSRODistillMetadata = _reflection.GeneratedProtocolMessageType(
    "PSRODistillMetadata",
    (_message.Message,),
    {
        "DESCRIPTOR": _PSRODISTILLMETADATA,
        "__module__": "manager_pb2"
        # @@protoc_insertion_point(class_scope:PSRODistillMetadata)
    },
)
_sym_db.RegisterMessage(PSRODistillMetadata)


_PSRODISTILLMANAGER = _descriptor.ServiceDescriptor(
    name="PSRODistillManager",
    full_name="PSRODistillManager",
    file=DESCRIPTOR,
    index=0,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_start=697,
    serialized_end=1099,
    methods=[
        _descriptor.MethodDescriptor(
            name="GetLogDir",
            full_name="PSRODistillManager.GetLogDir",
            index=0,
            containing_service=None,
            input_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
            output_type=_PSRODISTILLSTRING,
            serialized_options=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.MethodDescriptor(
            name="GetManagerMetaData",
            full_name="PSRODistillManager.GetManagerMetaData",
            index=1,
            containing_service=None,
            input_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
            output_type=_PSRODISTILLMETADATA,
            serialized_options=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.MethodDescriptor(
            name="ClaimNewActivePolicyForPlayer",
            full_name="PSRODistillManager.ClaimNewActivePolicyForPlayer",
            index=2,
            containing_service=None,
            input_type=_PSRODISTILLPLAYER,
            output_type=_PSRODISTILLNEWBESTRESPONSEPARAMS,
            serialized_options=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.MethodDescriptor(
            name="SubmitFinalBRPolicy",
            full_name="PSRODistillManager.SubmitFinalBRPolicy",
            index=3,
            containing_service=None,
            input_type=_PSRODISTILLPOLICYMETADATAREQUEST,
            output_type=_PSRODISTILLCONFIRMATION,
            serialized_options=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.MethodDescriptor(
            name="IsPolicyFixed",
            full_name="PSRODistillManager.IsPolicyFixed",
            index=4,
            containing_service=None,
            input_type=_PSRODISTILLPLAYERANDPOLICYNUM,
            output_type=_PSRODISTILLCONFIRMATION,
            serialized_options=None,
            create_key=_descriptor._internal_create_key,
        ),
    ],
)
_sym_db.RegisterServiceDescriptor(_PSRODISTILLMANAGER)

DESCRIPTOR.services_by_name["PSRODistillManager"] = _PSRODISTILLMANAGER

# @@protoc_insertion_point(module_scope)
