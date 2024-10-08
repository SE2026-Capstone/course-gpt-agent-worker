import {Worker} from "bullmq"
import {Redis} from "ioredis"
import "dotenv/config"

const REDIS_URL = process.env.REDIS_URL ?? "localhost:6379"
const WORKER_CONCURRENCY = Number.parseInt(process.env.WORKER_CONCURRENCY ?? "10")
const CHAT_JOB_QUEUE_NAME = process.env.CHAT_JOB_QUEUE_NAME ?? "chat-job"

const redisConnection = new Redis(REDIS_URL, {
  maxRetriesPerRequest: null,
})

const agentWorker = new Worker(CHAT_JOB_QUEUE_NAME, async (job) => {
  console.log(job?.data?.user_email)
  return 
}, {
  concurrency: WORKER_CONCURRENCY,
  autorun: false,
  connection: redisConnection
})

export default agentWorker