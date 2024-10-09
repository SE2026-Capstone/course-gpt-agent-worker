import agent, { GraphAnnotation } from "./agent"
import "dotenv/config"

// skip LLM tests if not enabled, this is to prevent API costs
const testIfAgentTestsEnabled = (process.env.ENABLE_AGENT_TESTS === "true") ? test : test.skip

testIfAgentTestsEnabled("normal user message", async () => {
    const initialState: typeof GraphAnnotation.State = {
        rawUserChat: "What courses are related to computer science?",
        retrievedDocuments: undefined,
        semanticSearchQuery: undefined,
        messages: []
    }
    const result = await agent.invoke(initialState)
    
})