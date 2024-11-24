import { GraphAnnotation } from "./agent/agent"
import { OpenAIEmbeddings } from "@langchain/openai";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import path from 'path';
import dotenv from 'dotenv';

// Load environment variables from .env file
dotenv.config({ path: path.resolve(__dirname, '../.env') });

const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
    apiKey: process.env.OPENAI_API_KEY || ""
});

const vectorStore = new Chroma(embeddings, {
  collectionName: "uw-courses-test-4",
  url: "http://localhost:8000",
  collectionMetadata: {
      "hnsw:space": "cosine",
  },
});

export const vectorSimilaritySearch = async (state: typeof GraphAnnotation.State) => {
    console.log("vectorSimilaritySearch");

    const documents = await vectorStore.similaritySearchWithScore(state.semanticSearchQuery || state.rawUserChat, 10, );
    // return the top 5 documents with the highest score 
    if (documents.length > 0) {
      const sortedDocuments = documents.sort((a, b) => b[1] - a[1]);
      return sortedDocuments.map((doc) => doc[0]).slice(0, 5);
    }

    console.log("inside vectorSimilaritySearch: ", documents);

    return [];
};

// hybrid search 

// term based match making 