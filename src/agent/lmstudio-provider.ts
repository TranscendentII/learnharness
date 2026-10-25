/**
 * LM Studio's OpenAI-compatible local server.
 *
 * Keyless by default. Set LMSTUDIO_API_KEY only when server authentication
 * is enabled in LM Studio. The default matches the port used by this setup;
 * override LMSTUDIO_BASE_URL if your LM Studio server uses another port.
 */
import { createProvider, type Model } from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";

export const LMSTUDIO_PROVIDER_ID = "lmstudio";
export const LMSTUDIO_DEFAULT_BASE_URL = "http://127.0.0.1:1235/v1";

type ModelRecord = Record<string, unknown>;

export function lmStudioBaseUrl(): string {
	const raw = process.env.LMSTUDIO_BASE_URL?.trim() || LMSTUDIO_DEFAULT_BASE_URL;
	const normalized = raw.replace(/\/+$/, "");
	return /\/v1$/i.test(normalized) ? normalized : `${normalized}/v1`;
}

function endpointUrl(baseUrl: string, pathname: string): string {
	const url = new URL(baseUrl);
	url.pathname = pathname;
	url.search = "";
	url.hash = "";
	return url.toString();
}

function asRecord(value: unknown): ModelRecord | undefined {
	return value !== null && typeof value === "object" && !Array.isArray(value)
		? (value as ModelRecord)
		: undefined;
}

function recordsFrom(payload: unknown): ModelRecord[] {
	const root = asRecord(payload);
	if (!root) return [];
	const entries = Array.isArray(root.data) ? root.data : Array.isArray(root.models) ? root.models : [];
	return entries.map(asRecord).filter((item): item is ModelRecord => item !== undefined);
}

function positiveInteger(value: unknown, fallback: number): number {
	const parsed = typeof value === "number" ? value : Number(value);
	return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : fallback;
}

export function lmStudioModelFromInfo(
	info: ModelRecord,
	baseUrl = lmStudioBaseUrl(),
): Model<"openai-completions"> | undefined {
	const id = typeof info.id === "string" ? info.id.trim() : "";
	if (!id) return undefined;
	const type = typeof info.type === "string" ? info.type.toLowerCase() : "";
	if (type && type !== "llm" && type !== "model") return undefined;
	const capabilities = Array.isArray(info.capabilities)
		? info.capabilities.filter((item): item is string => typeof item === "string")
		: [];
	const contextWindow = positiveInteger(info.max_context_length ?? info.context_length, 32768);
	const maxTokens = Math.min(
		positiveInteger(info.max_output_tokens, Math.min(contextWindow, 8192)),
		contextWindow,
	);
	const displayName = info.display_name ?? info.name;
	const name = typeof displayName === "string" && displayName.trim() ? displayName.trim() : id;
	const supportsVision = capabilities.some((capability) => /vision|image/i.test(capability));
	const supportsReasoning =
		capabilities.some((capability) => /reasoning|thinking/i.test(capability)) ||
		/(deepseek-r1|\bqwq\b|reasoner|thinking)/i.test(id);
	return {
		id,
		name,
		api: "openai-completions",
		provider: LMSTUDIO_PROVIDER_ID,
		baseUrl,
		reasoning: supportsReasoning,
		input: supportsVision ? ["text", "image"] : ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow,
		maxTokens,
	};
}

export function lmStudioModelForId(id: string): Model<"openai-completions"> {
	const trimmed = id.trim();
	if (!trimmed) throw new Error("LM Studio model id is empty");
	const model = lmStudioModelFromInfo({ id: trimmed }, lmStudioBaseUrl());
	if (!model) throw new Error(`Could not construct LM Studio model: ${trimmed}`);
	return model;
}

export function createLmStudioProvider() {
	const baseUrl = lmStudioBaseUrl();
	return createProvider<"openai-completions">({
		id: LMSTUDIO_PROVIDER_ID,
		name: "LM Studio",
		baseUrl,
		auth: {
			apiKey: {
				name: "LM Studio",
				resolve: async () => {
					const apiKey = process.env.LMSTUDIO_API_KEY?.trim();
					return { auth: apiKey ? { apiKey } : {} };
				},
			},
		},
		models: [],
		api: openAICompletionsApi(),
		fetchModels: async ({ signal }) => {
			const headers: Record<string, string> = { Accept: "application/json" };
			const apiKey = process.env.LMSTUDIO_API_KEY?.trim();
			if (apiKey) headers.Authorization = `Bearer ${apiKey}`;
			const response = await fetch(endpointUrl(baseUrl, "/v1/models"), { headers, signal });
			if (!response.ok) {
				throw new Error(`LM Studio model discovery failed (HTTP ${response.status}). Start the LM Studio server and load a chat model.`);
			}
			const available = recordsFrom(await response.json());
			if (available.length === 0) return [];
			const metadata = new Map<string, ModelRecord>();
			try {
				const detailsResponse = await fetch(endpointUrl(baseUrl, "/api/v0/models"), { headers, signal });
				if (detailsResponse.ok) {
					for (const info of recordsFrom(await detailsResponse.json())) {
						if (typeof info.id === "string") metadata.set(info.id, info);
					}
				}
			} catch (error) {
				if (signal.aborted) throw error;
			}
			return available
				.map((listed) => {
					const id = typeof listed.id === "string" ? listed.id : "";
					return lmStudioModelFromInfo({ ...listed, ...(metadata.get(id) ?? {}), id }, baseUrl);
				})
				.filter((model): model is Model<"openai-completions"> => model !== undefined);
		},
	});
}
