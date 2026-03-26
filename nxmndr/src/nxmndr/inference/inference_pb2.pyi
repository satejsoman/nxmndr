from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ModelFormat(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MODEL_FORMAT_UNSPECIFIED: _ClassVar[ModelFormat]
    PYTORCH: _ClassVar[ModelFormat]
    ONNX: _ClassVar[ModelFormat]
    HUGGINGFACE: _ClassVar[ModelFormat]
    TORCHHUB: _ClassVar[ModelFormat]

class TaskType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TASK_TYPE_UNSPECIFIED: _ClassVar[TaskType]
    CLASSIFICATION: _ClassVar[TaskType]
    SEGMENTATION: _ClassVar[TaskType]
    OBJECT_DETECTION: _ClassVar[TaskType]
    EMBEDDING: _ClassVar[TaskType]
    CUSTOM: _ClassVar[TaskType]
MODEL_FORMAT_UNSPECIFIED: ModelFormat
PYTORCH: ModelFormat
ONNX: ModelFormat
HUGGINGFACE: ModelFormat
TORCHHUB: ModelFormat
TASK_TYPE_UNSPECIFIED: TaskType
CLASSIFICATION: TaskType
SEGMENTATION: TaskType
OBJECT_DETECTION: TaskType
EMBEDDING: TaskType
CUSTOM: TaskType

class MetadataEntry(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class TransformSpec(_message.Message):
    __slots__ = ("name", "params")
    class ParamsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    name: str
    params: _containers.ScalarMap[str, str]
    def __init__(self, name: _Optional[str] = ..., params: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ModelSpec(_message.Message):
    __slots__ = ("model_id", "format", "name", "source", "model_class", "task", "preprocessing", "postprocessing", "metadata", "checksum", "version", "artifact_mime_type", "artifact", "lazy_load")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    MODEL_CLASS_FIELD_NUMBER: _ClassVar[int]
    TASK_FIELD_NUMBER: _ClassVar[int]
    PREPROCESSING_FIELD_NUMBER: _ClassVar[int]
    POSTPROCESSING_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    CHECKSUM_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_MIME_TYPE_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_FIELD_NUMBER: _ClassVar[int]
    LAZY_LOAD_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    format: ModelFormat
    name: str
    source: str
    model_class: str
    task: TaskType
    preprocessing: _containers.RepeatedCompositeFieldContainer[TransformSpec]
    postprocessing: _containers.RepeatedCompositeFieldContainer[TransformSpec]
    metadata: _containers.RepeatedCompositeFieldContainer[MetadataEntry]
    checksum: str
    version: str
    artifact_mime_type: str
    artifact: bytes
    lazy_load: bool
    def __init__(self, model_id: _Optional[str] = ..., format: _Optional[_Union[ModelFormat, str]] = ..., name: _Optional[str] = ..., source: _Optional[str] = ..., model_class: _Optional[str] = ..., task: _Optional[_Union[TaskType, str]] = ..., preprocessing: _Optional[_Iterable[_Union[TransformSpec, _Mapping]]] = ..., postprocessing: _Optional[_Iterable[_Union[TransformSpec, _Mapping]]] = ..., metadata: _Optional[_Iterable[_Union[MetadataEntry, _Mapping]]] = ..., checksum: _Optional[str] = ..., version: _Optional[str] = ..., artifact_mime_type: _Optional[str] = ..., artifact: _Optional[bytes] = ..., lazy_load: bool = ...) -> None: ...

class LoadModelRequest(_message.Message):
    __slots__ = ("spec", "overwrite")
    SPEC_FIELD_NUMBER: _ClassVar[int]
    OVERWRITE_FIELD_NUMBER: _ClassVar[int]
    spec: ModelSpec
    overwrite: bool
    def __init__(self, spec: _Optional[_Union[ModelSpec, _Mapping]] = ..., overwrite: bool = ...) -> None: ...

class LoadModelResponse(_message.Message):
    __slots__ = ("success", "model_id", "message", "effective_metadata")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    EFFECTIVE_METADATA_FIELD_NUMBER: _ClassVar[int]
    success: bool
    model_id: str
    message: str
    effective_metadata: _containers.RepeatedCompositeFieldContainer[MetadataEntry]
    def __init__(self, success: bool = ..., model_id: _Optional[str] = ..., message: _Optional[str] = ..., effective_metadata: _Optional[_Iterable[_Union[MetadataEntry, _Mapping]]] = ...) -> None: ...

class UnloadModelRequest(_message.Message):
    __slots__ = ("model_id",)
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    def __init__(self, model_id: _Optional[str] = ...) -> None: ...

class UnloadModelResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

class ListModelsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ModelInfo(_message.Message):
    __slots__ = ("model_id", "name", "format", "task", "device", "loaded")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    TASK_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    LOADED_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    name: str
    format: ModelFormat
    task: TaskType
    device: str
    loaded: bool
    def __init__(self, model_id: _Optional[str] = ..., name: _Optional[str] = ..., format: _Optional[_Union[ModelFormat, str]] = ..., task: _Optional[_Union[TaskType, str]] = ..., device: _Optional[str] = ..., loaded: bool = ...) -> None: ...

class ListModelsResponse(_message.Message):
    __slots__ = ("models",)
    MODELS_FIELD_NUMBER: _ClassVar[int]
    models: _containers.RepeatedCompositeFieldContainer[ModelInfo]
    def __init__(self, models: _Optional[_Iterable[_Union[ModelInfo, _Mapping]]] = ...) -> None: ...

class ListModelRegistryRequest(_message.Message):
    __slots__ = ("provider_id",)
    PROVIDER_ID_FIELD_NUMBER: _ClassVar[int]
    provider_id: str
    def __init__(self, provider_id: _Optional[str] = ...) -> None: ...

class ModelRegistryEntry(_message.Message):
    __slots__ = ("entry_id", "model_id", "display_name", "source", "task", "format", "last_used_epoch_ms", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ENTRY_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    DISPLAY_NAME_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    TASK_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    LAST_USED_EPOCH_MS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    entry_id: str
    model_id: str
    display_name: str
    source: str
    task: str
    format: str
    last_used_epoch_ms: int
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, entry_id: _Optional[str] = ..., model_id: _Optional[str] = ..., display_name: _Optional[str] = ..., source: _Optional[str] = ..., task: _Optional[str] = ..., format: _Optional[str] = ..., last_used_epoch_ms: _Optional[int] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ListModelRegistryResponse(_message.Message):
    __slots__ = ("entries",)
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    entries: _containers.RepeatedCompositeFieldContainer[ModelRegistryEntry]
    def __init__(self, entries: _Optional[_Iterable[_Union[ModelRegistryEntry, _Mapping]]] = ...) -> None: ...

class EvictModelRegistryRequest(_message.Message):
    __slots__ = ("entry_id", "provider_id")
    ENTRY_ID_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_ID_FIELD_NUMBER: _ClassVar[int]
    entry_id: str
    provider_id: str
    def __init__(self, entry_id: _Optional[str] = ..., provider_id: _Optional[str] = ...) -> None: ...

class EvictModelRegistryResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

class PredictRequest(_message.Message):
    __slots__ = ("model_id", "input", "shape", "dtype", "options")
    class OptionsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    input: bytes
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    options: _containers.ScalarMap[str, str]
    def __init__(self, model_id: _Optional[str] = ..., input: _Optional[bytes] = ..., shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., options: _Optional[_Mapping[str, str]] = ...) -> None: ...

class PredictResponse(_message.Message):
    __slots__ = ("output", "shape", "dtype", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    output: bytes
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, output: _Optional[bytes] = ..., shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class StreamPredictRequest(_message.Message):
    __slots__ = ("model_id", "chunk", "shape", "dtype", "end_of_sequence", "context")
    class ContextEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    CHUNK_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    END_OF_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    chunk: bytes
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    end_of_sequence: bool
    context: _containers.ScalarMap[str, str]
    def __init__(self, model_id: _Optional[str] = ..., chunk: _Optional[bytes] = ..., shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., end_of_sequence: bool = ..., context: _Optional[_Mapping[str, str]] = ...) -> None: ...

class StreamPredictResponse(_message.Message):
    __slots__ = ("output", "shape", "dtype", "end_of_sequence", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    END_OF_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    output: bytes
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    end_of_sequence: bool
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, output: _Optional[bytes] = ..., shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., end_of_sequence: bool = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class CapabilitiesRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class CapabilityInfo(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class CapabilitiesResponse(_message.Message):
    __slots__ = ("capabilities",)
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    capabilities: _containers.RepeatedCompositeFieldContainer[CapabilityInfo]
    def __init__(self, capabilities: _Optional[_Iterable[_Union[CapabilityInfo, _Mapping]]] = ...) -> None: ...

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("ready", "message")
    READY_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ready: bool
    message: str
    def __init__(self, ready: bool = ..., message: _Optional[str] = ...) -> None: ...

class ErrorStatus(_message.Message):
    __slots__ = ("code", "message", "details")
    CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    code: int
    message: str
    details: str
    def __init__(self, code: _Optional[int] = ..., message: _Optional[str] = ..., details: _Optional[str] = ...) -> None: ...
