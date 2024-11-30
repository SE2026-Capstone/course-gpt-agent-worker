import path from "path";
import dotenv from "dotenv";

import agent, { GraphAnnotation } from "./agent/agent"

// Load environment variables
dotenv.config({ path: path.resolve(__dirname, "../.env") });



async function testIfAgentTestsEnabled() {
    const initialState: typeof GraphAnnotation.State = {
        rawUserChat: "What is a cool introductory Artificial Intelligence course?",
        retrievedDocuments: undefined,
        semanticSearchQuery: undefined,
        messages: [],
        answer: ""
    }
    const result = await agent.invoke(initialState)
    
}

testIfAgentTestsEnabled();
