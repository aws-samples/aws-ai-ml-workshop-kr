export interface ReviewBotProps {
    repoType: string;
    awsRegion: string;
    bedrockModel: string;
    maxTokens?: number;
    temperature?: number;
    slackChannel: string;
    slackNotification: string;
  }
  
  export interface LambdaEnvironment {
    POWERTOOLS_SERVICE_NAME: string;
    LOG_LEVEL: string;
  }
  