CREATE TABLE dbo.TranslationJobHeader
(
    JobId UNIQUEIDENTIFIER NOT NULL
        CONSTRAINT PK_TranslationJobHeader
        PRIMARY KEY
        DEFAULT NEWSEQUENTIALID(),

    UserId NVARCHAR(100) NOT NULL,

    OriginalFileName NVARCHAR(500) NOT NULL,

    FileExtension NVARCHAR(20) NOT NULL,

    SourceLanguage NVARCHAR(50) NOT NULL,

    TargetLanguage NVARCHAR(50) NOT NULL,

    OriginalObjectKey NVARCHAR(1000) NOT NULL,

    OutputObjectKey NVARCHAR(1000) NULL,

    OverallStatus NVARCHAR(30) NOT NULL
        CONSTRAINT DF_TranslationJobHeader_Status
        DEFAULT N'UPLOADED',

    CreatedAt DATETIME2(3) NOT NULL
        CONSTRAINT DF_TranslationJobHeader_CreatedAt
        DEFAULT SYSUTCDATETIME(),

    CompletedAt DATETIME2(3) NULL,

    CONSTRAINT CK_TranslationJobHeader_Status
    CHECK
    (
        OverallStatus IN
        (
            N'UPLOADED',
            N'PROCESSING',
            N'COMPLETED',
            N'FAILED',
            N'CANCELLED'
        )
    )
);
GO