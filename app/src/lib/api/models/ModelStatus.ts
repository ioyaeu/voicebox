/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Response model for model status.
 */
export type ModelStatus = {
  model_name: string;
  display_name: string;
  engine: string;
  hf_repo_id?: string | null;
  model_size?: string;
  languages?: Array<string>;
  downloaded: boolean;
  downloading?: boolean; // True if download is in progress
  size_mb?: number | null;
  loaded?: boolean;
};
