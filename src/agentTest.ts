import path from "path";
import dotenv from "dotenv";

import agent, { GraphAnnotation } from "./agent/agent"

// Load environment variables
dotenv.config({ path: path.resolve(__dirname, "../.env") });



async function testIfAgentTestsEnabled() {
    // const queryResult = await vectorStore.similaritySearch("Artificial Intelligence courses in the CS department offered in third year", 5);

    // console.log(queryResult);

    const initialState: typeof GraphAnnotation.State = {
        rawUserChat: "What is a cool introductory Artificial Intelligence course?",
        retrievedDocuments: undefined,
        semanticSearchQuery: undefined,
        messages: [],
        answers: []
    }
    const result = await agent.invoke(initialState)
    
}

testIfAgentTestsEnabled();
