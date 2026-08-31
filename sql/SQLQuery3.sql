CREATE TABLE dbo.TranslationJobDetail
(
    DetailId UNIQUEIDENTIFIER NOT NULL
        CONSTRAINT PK_TranslationJobDetail
        PRIMARY KEY
        DEFAULT NEWSEQUENTIALID(),

    JobId UNIQUEIDENTIFIER NOT NULL,

    ProcessType NVARCHAR(60) NOT NULL,

    Status NVARCHAR(30) NOT NULL,

    StartedAt DATETIME2(3) NOT NULL,

    CompletedAt DATETIME2(3) NULL,

    DurationMs BIGINT NULL,

    RequestedModel NVARCHAR(255) NULL,

    FulfilledModel NVARCHAR(255) NULL,

    ModelVersion NVARCHAR(255) NULL,

    ProviderName NVARCHAR(255) NULL,

    RouteStrategy NVARCHAR(100) NULL,

    FallbackUsed BIT NULL,

    OpenRouterRequestId NVARCHAR(255) NULL,

    PromptTokens BIGINT NULL,

    CompletionTokens BIGINT NULL,

    TotalTokens BIGINT NULL,

    CachedTokens BIGINT NULL,

    CacheWriteTokens BIGINT NULL,

    ReasoningTokens BIGINT NULL,

    ImageCount INT NULL,

    ImageDetail NVARCHAR(30) NULL,

    InputBytes BIGINT NULL,

    OutputBytes BIGINT NULL,

    FinishReason NVARCHAR(100) NULL,

    HttpStatusCode INT NULL,

    EstimatedCostUsd DECIMAL(18,8) NULL,

    RequestMetadata NVARCHAR(MAX) NULL,

    ResponseMetadata NVARCHAR(MAX) NULL,

    Message NVARCHAR(MAX) NULL,

    ErrorDetails NVARCHAR(MAX) NULL,

    CreatedAt DATETIME2(3) NOT NULL
        CONSTRAINT DF_TranslationJobDetail_CreatedAt
        DEFAULT SYSUTCDATETIME(),

    CONSTRAINT FK_TranslationJobDetail_Header
        FOREIGN KEY (JobId)
        REFERENCES dbo.TranslationJobHeader(JobId),

    CONSTRAINT CK_TranslationJobDetail_ProcessType
    CHECK
    (
        ProcessType IN
        (
            N'UPLOAD',
            N'FILE_ANALYSIS',
            N'OCR',
            N'OCR_VERIFICATION',
            N'LLM_TEXT_TRANSLATION',
            N'LLM_IMAGE_ANALYSIS',
            N'LLM_IMAGE_TRANSLATION',
            N'RENDER',
            N'RECONSTRUCTION',
            N'VALIDATION',
            N'FINAL_OUTPUT'
        )
    ),

    CONSTRAINT CK_TranslationJobDetail_Status
    CHECK
    (
        Status IN
        (
            N'STARTED',
            N'COMPLETED',
            N'FAILED',
            N'SKIPPED'
        )
    )
);
GO