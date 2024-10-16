import { ChatOpenAI } from "@langchain/openai";
import { StateGraph, Annotation, MessagesAnnotation} from "@langchain/langgraph"
import {ChatPromptTemplate, FewShotChatMessagePromptTemplate, PromptTemplate} from "@langchain/core/prompts"
import {START, END} from "@langchain/langgraph"
import {z} from "zod"
import "dotenv/config"
import { vectorSimilaritySearch } from "../ragQueries"

const OPENAI_API_KEY = process.env.OPENAI_API_KEY

// Annotations are how graph state is represented in LangGraph
export const GraphAnnotation = Annotation.Root({
    ...MessagesAnnotation.spec,

    // users original message
    rawUserChat: Annotation<string>,

    // semantic search query
    semanticSearchQuery: Annotation<string|undefined>,

    // retrieved course documents from ChromaDB
    retrievedDocuments: Annotation<any>, // TODO: type this


})


// filter unrelated, inappropriate or incomprehensible messages
const initialFilteringEdge = async (state: typeof GraphAnnotation.State): Promise<string> => {
    console.log("initialFilteringEdge")
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
    const responseSchema = z.object({
        acceptable: z.boolean(),
        reason: z.string().optional()
    })
    const model = new ChatOpenAI({
        openAIApiKey: OPENAI_API_KEY,
        temperature: 0,
        modelName: "gpt-4o-mini"
    }).bind({
        response_format: { type: "json_object"}
    })
    const chain = prompt.pipe(model)

    const response = await chain.invoke({
        userMessage: state.rawUserChat
    })

    try {
        const asJSON = responseSchema.parse(JSON.parse(response.content.toString()))
        return "true"
    } catch (e: unknown) {
        if (e instanceof z.ZodError) {
            return "false"
        }
        return "false"
    }
}


// extract semantic search query
const semanticSearchQueryExtractionNode = async (state: typeof GraphAnnotation.State) => {
    console.log("semanticSearchQueryExtractionNode")
    const promptString = 
`You are an AI that helps extract semantic search queries from user messages.

You will be given a user message. Your task is to identify and extract key snippets that are directly relevant to the subject of the message. You will remove any verbs, commands, conversational dialogue, or unnecessary words from the original message. You will output one or more concise snippets that only contain key information. The output should read like a semantic search query.
For context, the user message will be related to university courses. Assume that you already know which university the user is interested in, so remove any text about the university itself.
Follow the rules in the rules section.

# Rules
- Remove action verbs or directives like "fetch", "show", "give me", "retrieve", etc
- Keep nouns and adjectives (except the university name)
- Prefer outputs that are sentences rather than just keywords. This means you need to strip away words from the original message, but not so much that the output is reduced to being just keywords
- Remove any text about the university itself, such as the name of a university (University of Waterloo, Harvard, MIT, university of Toronto, Cambridge University, etc)
- Remove the word "course" and its plural forms

# Output format

You will output a JSON object that adheres to the following format:
{{
    extracted_query: string,
    explanation: string
}}

The extracted_query field should contain the extracted query snippet taken from the original user message.
The explanation field should contain an explanation of the actions taken to extract the query. This explanation should be concise and to the point.
`
    const fewShotExamplePrompt = ChatPromptTemplate.fromMessages([
        ["human", `{userMessage}`],
        ["ai",
`{{
    extracted_query: "{idealAIResponse}",
    explanation: "{explanation}"
}}`]
    ])

    const fewShotExamples = [
        {
            userMessage: "Fetch me a list of UW courses that are related to machine learning or deep learning.",
            idealAIResponse: "machine learning or deep learning",
            explanation: "The verbs at the start of the sentence were removed, and the words 'UW courses' were removed."
        },
        {
            userMessage: "Can you give me a list of first-year computer science courses at the university of michigan?",
            idealAIResponse: "first year computer science",
            explanation: "The introductory phrase was removed, leaving only the relevant course information."
        },
        {
            userMessage: "Can you provide resources on natural language processing and deep learning at the University of Waterloo?",
            idealAIResponse: "natural language processing and deep learning",
            explanation: "The request phrase was removed, retaining only the topics of interest."
        },
        {
            userMessage: "What are the prerequisites for taking CS341?",
            idealAIResponse: "prerequisites for CS341",
            explanation: "The question format was simplified to focus on the prerequisites."
        },
        {
            userMessage: "Give me a list of third year courses about classical art and music. Sort the list of courses in alphabetical order",
            idealAIResponse: "third year classical art and music",
            explanation: "The action and sorting instructions were omitted, highlighting the relevant subjects."
        },
        {
            userMessage: "What courses are there at UPENN which deal with renaissance art?",
            idealAIResponse: "renaissance art",
            explanation: "The inquiry aspect was removed, concentrating on the specific subject matter."
        },
        {
            userMessage: "What courses are related to computer science?",
            idealAIResponse: "computer science",
            explanation: "The verbs at the start of the sentence were removed, and the words 'courses' were removed."
        }
    ]

    const fewShotPrompt = new FewShotChatMessagePromptTemplate({
        examplePrompt: fewShotExamplePrompt,
        examples: fewShotExamples,
        inputVariables: [],
    })

    const prompt = ChatPromptTemplate.fromMessages([
        ["system", promptString],
        // https://github.com/langchain-ai/langchainjs/issues/5331#issuecomment-2198319629
        ChatPromptTemplate.fromMessages((await fewShotPrompt.invoke({})).toChatMessages()),
        ["human", "{input}"]
    ])

    console.log((await prompt.invoke({input: state.rawUserChat})).toChatMessages())

    const response = await prompt.pipe(new ChatOpenAI({
        openAIApiKey: OPENAI_API_KEY,
        temperature: 0,
        modelName: "gpt-4o-mini"
    }).bind({
        response_format: { type: "json_object"}
    })).invoke({input: state.rawUserChat})


    const responseSchema = z.object({
        extracted_query: z.string(),
        explanation: z.string()
    })

    try {
        const result = responseSchema.parse(JSON.parse(response.content.toString()))
        return {
            semanticSearchQuery: result.extracted_query,
        }
    } catch (e) {
        console.log("error", e)
        return null
    }
}

const semanticSearchAndAnswer = async (state: typeof GraphAnnotation.State) => {
    console.log("semanticSearchAndAnswer")
    // RAG search on chromadb + filter 
    const documents  = await vectorSimilaritySearch(state);
    console.log("documents", documents);

    // add documents to langraph state 
    state.retrievedDocuments = documents;

    const context = documents.map((doc) => doc.pageContent).join("\n\n");

    const promptString = `
You are a helpful assistant that can answer questions about the context.
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question}
`

    // use prompt to get answer 
    const prompt = ChatPromptTemplate.fromMessages([
        ["system", promptString]
    ]);
    

    const response = await prompt.pipe(new ChatOpenAI({
        openAIApiKey: OPENAI_API_KEY,
        temperature: 0,
        modelName: "gpt-4o-mini"
    }).bind({
        response_format: { type: "json_object"}
    })).invoke({
        context: context,
        question: state.rawUserChat
    })

    const responseSchema = z.object({
        answer: z.string(),
    })

    try {
        const result = responseSchema.parse(JSON.parse(response.content.toString()))
        return {
            answer: result.answer,
        }
    } catch (e) {
        console.log("error", e)
        return null
    }
}


const decideCourseList = async (state: typeof GraphAnnotation.State) => {
    console.log("decideCourseList")
    return
}


const validateOutput = async (state: typeof GraphAnnotation.State) => {
    console.log("validateOutput")
    const promptString = 
`Your an AI that helps validate the output of the system.
Validate that the system output is of the form described below.
A JSON object that adheres to the following rules:

# Rules
{{
    extracted_query: string,
    explanation: string
}}

The extracted_query field should contain the extracted query snippet taken from the original user message.
The explanation field should contain an explanation of the actions taken to extract the query. This explanation should be concise and to the point.

# Output format
If the output is not valid, return "Invalid"
If the output is valid, return "Valid"
`
    return
}

// return json
const outputNode = async (state: typeof GraphAnnotation.State) => {
    console.log(JSON.stringify(state, null, 2))
    return
}

// graph compilation
const agentGraph = new StateGraph(GraphAnnotation)
    .addNode("semanticSearchQueryExtractionNode", semanticSearchQueryExtractionNode)
    .addNode("outputNode", outputNode)
    .addEdge("semanticSearchQueryExtractionNode", "outputNode")
    .addConditionalEdges(START, initialFilteringEdge, {
        true: "semanticSearchQueryExtractionNode",
        false: END
    })
    

const agent = agentGraph.compile()

export default agent