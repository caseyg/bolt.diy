import { BaseProvider } from '~/lib/modules/llm/base-provider';
import type { ModelInfo } from '~/lib/modules/llm/types';
import type { IProviderSetting } from '~/types/model';
import type { LanguageModelV1 } from 'ai';
import { createOpenAI } from '@ai-sdk/openai';
import { logger } from '~/utils/logger';
import { LLMManager } from '~/lib/modules/llm/manager';

export default class IBMwatsonxProvider extends BaseProvider {
  name = 'IBM watsonx';
  getApiKeyLink = 'https://cloud.ibm.com/watsonx/overview';
  labelForGetApiKey = 'Get IBM watsonx API Key';

  config = {
    baseUrlKey: 'IBM_WATSONX_API_BASE_URL',
    apiTokenKey: 'IBM_WATSONX_API_KEY',
    projectIdKey: 'IBM_WATSONX_PROJECT_ID',
    spaceIdKey: 'IBM_WATSONX_SPACE_ID',
    instanceCrnKey: 'IBM_WATSONX_INSTANCE_CRN',
  };

  // Cache for the IAM token
  private tokenCache: {
    token: string;
    expiresAt: number;
  } | null = null;

  staticModels: ModelInfo[] = [{
    name: 'mistralai/mistral-small-24b-instruct-2501',
    label: 'mistralai/mistral-small-24b-instruct-2501',
    provider: 'IBM watsonx',
    maxTokenAllowed: 32768,
  }];

  async getDynamicModels(
    apiKeys?: Record<string, string>,
    settings?: IProviderSetting,
    serverEnv: Record<string, string> = {},
  ): Promise<ModelInfo[]> {
    const { 
      baseUrl: fetchBaseUrl, 
      apiKey,
      projectId,
      spaceId,
      instanceCrn
    } = this.getProviderSettings({
      apiKeys,
      providerSettings: settings,
      serverEnv,
    });
    
    const baseUrl = fetchBaseUrl || 'https://us-south.ml.cloud.ibm.com';

    if (!apiKey) {
      throw `Missing API Key configuration for ${this.name} provider`;
    }
    
    // Check if we have at least one required ID
    if (!projectId && !spaceId && !instanceCrn) {
      logger.warn('IBM watsonx requires at least one of: project_id, space_id, or wml_instance_crn. Dynamic model loading may fail.');
    }

    try {
      // Get an IAM token first
      const token = await this.getIAMToken(apiKey);
      
      // Fetch from foundation model specs endpoint
      const foundationModelResponse = await fetch(
        `${baseUrl}/ml/v1/foundation_model_specs?version=2024-05-31&limit=200`, 
        {
          headers: {
            Authorization: `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
        }
      );

      if (!foundationModelResponse.ok) {
        logger.warn(`Error fetching IBM watsonx foundation models: ${foundationModelResponse.statusText}. Falling back to models endpoint.`);
      } else {
        const foundationModelData = await foundationModelResponse.json() as any;
        const staticModelIds = this.staticModels.map((m) => m.name);
        
        if (foundationModelData.resources && foundationModelData.resources.length > 0) {
          logger.info(`Found ${foundationModelData.resources.length} foundation models from IBM watsonx API`);
          
          // Transform foundation model data to ModelInfo format
          const foundationModels = foundationModelData.resources
            .filter((model: any) => !staticModelIds.includes(model.model_id))
            .map((model: any) => {
              // Determine max token allowed based on model size or defaults
              let maxTokenAllowed = 8192;
              
              // Very large models (70B+) typically have larger context windows
              if (model.number_params && 
                 (model.number_params.includes('70b') || 
                  model.number_params.includes('90b') || 
                  model.number_params.includes('405b'))) {
                maxTokenAllowed = 32768;
              }
              
              // Dedicated code models often have larger context windows
              if (model.tasks && model.tasks.some((task: any) => task.id === 'code')) {
                maxTokenAllowed = Math.max(maxTokenAllowed, 16384);
              }
              
              return {
                name: model.model_id,
                label: model.label || model.model_id,
                provider: this.name,
                maxTokenAllowed: maxTokenAllowed,
                // Include additional metadata that might be useful for the UI
                metadata: {
                  provider: model.provider,
                  source: model.source,
                  description: model.short_description,
                  parameterSize: model.number_params,
                  tasks: model.tasks?.map((task: any) => task.id)
                }
              };
            });
            
          return foundationModels;
        }
      }
      
      // Fall back to the original models endpoint if foundation models endpoint fails
      logger.info('Falling back to models endpoint for IBM watsonx');
      const response = await fetch(`${baseUrl}/ml/v1/models?version=2024-05-31`, {
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Error fetching IBM watsonx models: ${response.statusText}`);
      }

      const res = await response.json() as any;
      const staticModelIds = this.staticModels.map((m) => m.name);
      
      // Filter to only include models suitable for text generation
      const data = (res.resources || []).filter(
        (model: any) => 
          model.status === 'available' && 
          !staticModelIds.includes(model.id) &&
          model.metadata?.task_type?.includes('text-generation')
      );

      return data.map((m: any) => ({
        name: m.id,
        label: `${m.name || m.id}`,
        provider: this.name,
        maxTokenAllowed: m.metadata?.max_sequence_length || 4096,
      }));
    } catch (error) {
      logger.error('Error getting IBM watsonx models:', error);
      return [];
    }
  }

  async getIAMToken(apiKey: string): Promise<string> {
    // Check if we have a cached token that's still valid
    if (this.tokenCache && this.tokenCache.expiresAt > Date.now()) {
      return this.tokenCache.token;
    }

    // Otherwise get a new token
    try {
      const response = await fetch('https://iam.cloud.ibm.com/identity/token', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey=${apiKey}`,
      });

      if (!response.ok) {
        throw new Error(`Failed to get IBM IAM token: ${response.statusText}`);
      }

      const data = await response.json() as { access_token: string; expires_in: number };
      
      // Cache the token with an expiry time (a bit shorter than the actual expiry to be safe)
      this.tokenCache = {
        token: data.access_token,
        expiresAt: Date.now() + (data.expires_in * 1000) - 300000, // Expire 5 minutes early
      };
      
      return data.access_token;
    } catch (error) {
      logger.error('Error getting IBM IAM token:', error);
      throw new Error('Failed to authenticate with IBM watsonx');
    }
  }
  
  getProviderSettings({
    apiKeys,
    providerSettings,
    serverEnv,
  }: {
    apiKeys?: Record<string, string>,
    providerSettings?: IProviderSetting,
    serverEnv: Record<string, string>,
  }) {
    // Get the API key and base URL using the existing method
    const { baseUrl, apiKey } = this.getProviderBaseUrlAndKey({
      apiKeys,
      providerSettings,
      serverEnv,
      defaultBaseUrlKey: this.config.baseUrlKey,
      defaultApiTokenKey: this.config.apiTokenKey,
    });
    
    const manager = LLMManager.getInstance();
    
    // Get the project ID, space ID, and instance CRN from all available sources
    const projectId = 
      apiKeys?.[this.config.projectIdKey] || 
      serverEnv[this.config.projectIdKey] ||
      process?.env?.[this.config.projectIdKey] ||
      manager.env?.[this.config.projectIdKey];
      
    const spaceId = 
      apiKeys?.[this.config.spaceIdKey] || 
      serverEnv[this.config.spaceIdKey] ||
      process?.env?.[this.config.spaceIdKey] ||
      manager.env?.[this.config.spaceIdKey];
      
    const instanceCrn = 
      apiKeys?.[this.config.instanceCrnKey] || 
      serverEnv[this.config.instanceCrnKey] ||
      process?.env?.[this.config.instanceCrnKey] ||
      manager.env?.[this.config.instanceCrnKey];
    
    // Debug log to check if we're picking up the environment variables
    logger.debug(`IBM watsonx settings - projectId: ${projectId ? 'set' : 'not set'}, spaceId: ${spaceId ? 'set' : 'not set'}, instanceCrn: ${instanceCrn ? 'set' : 'not set'}, baseUrl: ${baseUrl}, apiKey: ${apiKey ? 'set' : 'not set'}`);
    
    return { baseUrl, apiKey, projectId, spaceId, instanceCrn };
  }

  getModelInstance(options: {
    model: string;
    serverEnv: Env;
    apiKeys?: Record<string, string>;
    providerSettings?: Record<string, IProviderSetting>;
  }): LanguageModelV1 {
    const { model, serverEnv, apiKeys, providerSettings } = options;

    const { 
      baseUrl: fetchBaseUrl, 
      apiKey,
      projectId,
      spaceId,
      instanceCrn
    } = this.getProviderSettings({
      apiKeys,
      providerSettings: providerSettings?.[this.name],
      serverEnv: serverEnv as any,
    });
    
    if (!apiKey) {
      throw new Error(`Missing API key for ${this.name} provider`);
    }
    
    // Log environment variable values to help with debugging
    logger.info(`IBM watsonx credentials - project_id: ${projectId ? 'found' : 'missing'}, space_id: ${spaceId ? 'found' : 'missing'}, instance_crn: ${instanceCrn ? 'found' : 'missing'}`);
    
    // Check if we have at least one required ID
    if (!projectId && !spaceId && !instanceCrn) {
      // Instead of throwing an error, return a mock model that will return a friendly error message
      // This prevents the application from crashing when credentials are missing
      const mockOpenAI = createOpenAI({
        baseURL: 'https://api.openai.com/v1',
        apiKey: 'dummy-key',
        fetch: async () => {
          return new Response(
            JSON.stringify({
              error: {
                message: "IBM watsonx requires at least one of: project_id, space_id, or wml_instance_crn. Please configure these in your environment variables or API settings.",
                type: "invalid_request_error",
                code: "invalid_configuration"
              }
            }),
            { 
              status: 400, 
              headers: { 'Content-Type': 'application/json' } 
            }
          );
        }
      });
      
      logger.warn(`IBM watsonx requires at least one of: project_id, space_id, or wml_instance_crn. Model will return error messages.`);
      return mockOpenAI(model);
    }

    const baseUrl = fetchBaseUrl || 'https://us-south.ml.cloud.ibm.com';
    
    // Create a custom fetch function that handles IBM watsonx authentication and request formatting
    const customFetch = async (url: RequestInfo | URL, init?: RequestInit) => {
      try {
        // Get an IAM token
        const token = await this.getIAMToken(apiKey);
        
        // Create a new request with the token
        const requestInit: RequestInit = {
          ...init,
          headers: {
            ...init?.headers,
            'Authorization': `Bearer ${token}`,
          },
        };

        // Fix URL by removing the duplicate path
        let requestUrl = url.toString();
        if (requestUrl.includes('/chat/completions')) {
          requestUrl = `${baseUrl}/ml/v1/text/chat?version=2024-05-31`;
        }
        
        // Modify request body for IBM watsonx format
        if (init?.body) {
          try {
            const body = JSON.parse(init.body.toString());
            
            // Build the IBM watsonx request payload
            const watsonBody: any = {
              model_id: model,
              messages: body.messages.map((msg: any) => {
                if (typeof msg.content === 'string') {
                  return {
                    role: msg.role,
                    content: [{ type: 'text', text: msg.content }]
                  };
                }
                return msg;
              }),
              stream: body.stream
            };
            
            // Add the required IBM watsonx parameters
            if (projectId) {
              watsonBody.project_id = projectId;
            }
            
            if (spaceId) {
              watsonBody.space_id = spaceId;
            }
            
            if (instanceCrn) {
              watsonBody.wml_instance_crn = instanceCrn;
            }
            
            // Add other parameters that might be useful
            if (body.temperature !== undefined) {
              watsonBody.parameters = {
                ...(watsonBody.parameters || {}),
                temperature: body.temperature
              };
            }
            
            if (body.max_tokens !== undefined) {
              watsonBody.parameters = {
                ...(watsonBody.parameters || {}),
                max_new_tokens: body.max_tokens
              };
            }
            
            if (body.top_p !== undefined) {
              watsonBody.parameters = {
                ...(watsonBody.parameters || {}),
                top_p: body.top_p
              };
            }
            
            requestInit.body = JSON.stringify(watsonBody);
            
            // Debug logging to help troubleshoot
            logger.debug('IBM watsonx request payload:', JSON.stringify(watsonBody, null, 2));
          } catch (e) {
            logger.error('Error parsing request body:', e);
          }
        }
        
        // Make the request
        const response = await fetch(requestUrl, requestInit);
        
        if (!response.ok) {
          // Log the error response body to help with debugging
          const errorText = await response.text();
          logger.error('IBM watsonx API error response:', errorText);
          
          // Return the original response for error handling
          return new Response(errorText, {
            status: response.status,
            statusText: response.statusText,
            headers: response.headers
          });
        }
        
        return response;
      } catch (error) {
        logger.error('Error in IBM watsonx customFetch:', error);
        throw error;
      }
    };
    
    // Use OpenAI SDK with our custom fetch implementation
    const openai = createOpenAI({
      baseURL: 'https://api.openai.com/v1',
      apiKey: 'dummy-key', // This won't be used due to our custom fetch
      fetch: customFetch as any,
    });

    return openai(model);
  }
}