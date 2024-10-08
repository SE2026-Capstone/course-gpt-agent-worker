import { ChatOpenAI } from "@langchain/openai";
import { StateGraph, Annotation, MessagesAnnotation} from "@langchain/langgraph"
import {ChatPromptTemplate} from "@langchain/core/prompts"
import {START, END} from "@langchain/langgraph"
import "dotenv/config"

const OPENAI_API_KEY = process.env.OPENAI_API_KEY



export const GraphAnnotation = Annotation.Root({
    ...MessagesAnnotation.spec,

    // users original message
    rawUserChat: Annotation<string>,

    // retrieved course documents from ChromaDB
    retrievedDocuments: Annotation<any>, // TODO: type this


})


// filter unrelated, inappropriate or incomprehensible messages
const initialFilteringEdge = async (state: typeof GraphAnnotation.State): Promise<string> => {
    const prompt = ChatPromptTemplate.fromTemplate(
`Your task is to analyze a user message and determine if the message satisfies the following criteria:

1) The message is comprehensible
2) The message does not contain professionally inappropriate language
3) The message subject is related to courses at the University of Waterloo

You will output a JSON object that adheres to the following TypeScript interface:

{{
    acceptable: boolean,
    reason: string
}}

You will set the acceptable field to true if the message satisfies the criteria listed earlier.
Otherwise, set the acceptable field to false.
In either case, set the reason field to be a string explaining why the message is or is not acceptable.
Ensure that the reason string is short and to the point.

The user message is given below:
{userMessage}
`
    )
    const model = new ChatOpenAI({openAIApiKey: OPENAI_API_KEY, temperature: 0})
    const chain = prompt.pipe(model)

    const response = await chain.invoke({
        userMessage: state.rawUserChat
    })

    console.log(response)

    return "false"
}


// extract semantic search query
const semanticSearchQueryExtractionNode = (state: typeof GraphAnnotation.State) => {
    
}

// RAG search and answer

// decide if a list of courses would enhance the system response

// return json

// graph compilation
const agentGraph = new StateGraph(GraphAnnotation)
    .addConditionalEdges(START, initialFilteringEdge, {
        true: END,
        false: END
    })

const agent = agentGraph.compile()

export default agent