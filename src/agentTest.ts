import { Chroma } from "@langchain/community/vectorstores/chroma";
import { OpenAIEmbeddings } from "@langchain/openai";
import path from "path";
import dotenv from "dotenv";

import agent, { GraphAnnotation } from "./agent/agent"

// Load environment variables
dotenv.config({ path: path.resolve(__dirname, "../.env") });

// Initialize the embeddings
const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
    apiKey: process.env.OPENAI_API_KEY || "",
});

// Initialize Chroma vector store
const vectorStore = new Chroma(embeddings, {
    collectionName: "uw-courses-test-4",
    url: "http://localhost:8000",
    collectionMetadata: {
        "hnsw:space": "cosine",
    },
});


async function testIfAgentTestsEnabled() {
    // const queryResult = await vectorStore.similaritySearch("Artificial Intelligence courses in the CS department offered in third year", 5);

    // console.log(queryResult);

    const initialState: typeof GraphAnnotation.State = {
        rawUserChat: "What is a cool introductory Artificial Intelligence course?",
        retrievedDocuments: undefined,
        semanticSearchQuery: undefined,
        messages: []
    }
    const result = await agent.invoke(initialState)
    
}

testIfAgentTestsEnabled();
