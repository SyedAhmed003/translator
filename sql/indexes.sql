CREATE INDEX IX_TranslationJobHeader_UserId
ON dbo.TranslationJobHeader(UserId);
GO

CREATE INDEX IX_TranslationJobHeader_CreatedAt
ON dbo.TranslationJobHeader(CreatedAt DESC);
GO

CREATE INDEX IX_TranslationJobHeader_Status
ON dbo.TranslationJobHeader(OverallStatus);
GO

CREATE INDEX IX_TranslationJobDetail_JobId
ON dbo.TranslationJobDetail(JobId);
GO

CREATE INDEX IX_TranslationJobDetail_ProcessType
ON dbo.TranslationJobDetail(ProcessType);
GO

CREATE INDEX IX_TranslationJobDetail_StartedAt
ON dbo.TranslationJobDetail(StartedAt DESC);
GO

CREATE INDEX IX_TranslationJobDetail_OpenRouterRequestId
ON dbo.TranslationJobDetail(OpenRouterRequestId);
GO