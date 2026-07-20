class ServiceError(Exception):
    code = "internal_error"
    status_code = 500

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidRequestError(ServiceError):
    code = "invalid_request"
    status_code = 400


class ModelNotReadyError(ServiceError):
    code = "model_not_ready"
    status_code = 503


class InferenceTimeoutError(ServiceError):
    code = "inference_timeout"
    status_code = 504


class InferenceQueueFullError(ServiceError):
    code = "inference_queue_full"
    status_code = 503


class InferenceQueueTimeoutError(ServiceError):
    code = "inference_queue_timeout"
    status_code = 504


class InferenceError(ServiceError):
    code = "inference_failed"
    status_code = 500
