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


export const vectorSimilaritySearch = async (state: typeof GraphAnnotation.State) => {
    console.log("vectorSimilaritySearch");

    const documents = await vectorStore.similaritySearchWithScore(state.semanticSearchQuery || state.rawUserChat, 10, );

    // return the top 5 documents with the highest score 
    if (documents.length > 0) {
      const sortedDocuments = documents.sort((a, b) => b[1] - a[1]);
      return sortedDocuments.map((doc) => doc[0]).slice(0, 5);
    }

    return [];
};

// hybrid search 

// term based match making 