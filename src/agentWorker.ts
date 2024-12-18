import { Worker } from "bullmq"
import { Redis } from "ioredis"
import "dotenv/config"
import agent, { GraphAnnotation } from "./agent/agent"
import { AgentResponse } from "./types/agentResponse"

const REDIS_URL = process.env.REDIS_URL ?? "localhost:6379"
const WORKER_CONCURRENCY = Number.parseInt(process.env.WORKER_CONCURRENCY ?? "10")
const CHAT_JOB_QUEUE_NAME = process.env.CHAT_JOB_QUEUE_NAME ?? "chat-job"

const redisConnection = new Redis(REDIS_URL, {
	maxRetriesPerRequest: null,
})

const agentWorker = new Worker(CHAT_JOB_QUEUE_NAME, async (job) => {

	// run the graph agent
	const initialState: typeof GraphAnnotation.State = {
        rawUserChat: job.data,
        retrievedDocuments: [],
        semanticSearchQuery: undefined,
        messages: [],
		answer: ""
    }
    const result = await agent.invoke(initialState)

	console.log(result)

	const response: AgentResponse = {
		chat: result.answer,
		courseList: result.retrievedDocuments
	}

	// write values to redis
	return response;  
}, {
	concurrency: WORKER_CONCURRENCY,
	autorun: false,
	connection: redisConnection
})

export default agentWorker