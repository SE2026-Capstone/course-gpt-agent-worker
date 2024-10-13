import { GraphAnnotation } from "./agent/agent"
import { OpenAIEmbeddings } from "@langchain/openai";
import { Chroma } from "@langchain/community/vectorstores/chroma";

const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
    apiKey: process.env.OPENAI_API_KEY || ""
});

const vectorStore = new Chroma(embeddings, {
    collectionName: "uw-courses-test",
    url: "http://localhost:8080", // Optional, will default to this value
    collectionMetadata: {
      "hnsw:space": "cosine",
    }, // Optional, can be used to specify the distance method of the embedding space https://docs.trychroma.com/usage-guide#changing-the-distance-function
  });


// Thoughts: can probably turn this into a retreiver: https://js.langchain.com/docs/integrations/vectorstores/chroma/#query-by-turning-into-retriever
// vector similarity search 
export const vectorSimilaritySearch = async (state: typeof GraphAnnotation.State) => {
    console.log("vectorSimilaritySearch");

    // perform vector similarity search
    // vectorStore.similaritySearchWithScore can be used to get scores if we want to filter by score
    const documents = await vectorStore.similaritySearch(state.semanticSearchQuery || state.rawUserChat, 3);
    
    // check if there are documents
    if (documents.length === 0) {
        console.log('No matching documents found.');
        return { context: '', documents: [] };
    }

    // combine the context from matching documents
    const context = documents.map((doc) => doc.pageContent).join('\n\n - -\n\n');

    return { context, documents };
};

// hybrid search 

// term based match making 