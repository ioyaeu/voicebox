/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $ModelStatus = {
  description: `Response model for model status.`,
  properties: {
    model_name: {
      type: 'string',
      isRequired: true,
    },
    display_name: {
      type: 'string',
      isRequired: true,
    },
    engine: {
      type: 'string',
      isRequired: true,
    },
    hf_repo_id: {
      type: 'any-of',
      contains: [
        {
          type: 'string',
        },
        {
          type: 'null',
        },
      ],
    },
    model_size: {
      type: 'string',
    },
    languages: {
      type: 'array',
      contains: {
        type: 'string',
      },
    },
    downloaded: {
      type: 'boolean',
      isRequired: true,
    },
    size_mb: {
      type: 'any-of',
      contains: [
        {
          type: 'number',
        },
        {
          type: 'null',
        },
      ],
    },
    loaded: {
      type: 'boolean',
    },
  },
} as const;
