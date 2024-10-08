import agent, { GraphAnnotation } from "./agent"

test("normal user message", async () => {
    const initialState: typeof GraphAnnotation.State = {
        rawUserChat: "What courses are related to computer science?",
        retrievedDocuments: undefined,
        messages: []
    }
    const result = await agent.invoke(initialState)
    
})