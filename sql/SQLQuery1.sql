USE master;
GO

IF DB_ID(N'DocumentTranslatorDB') IS NULL
BEGIN
    CREATE DATABASE DocumentTranslatorDB;
END
GO

USE DocumentTranslatorDB;
GO
SELECT DB_NAME() AS CurrentDatabase;
