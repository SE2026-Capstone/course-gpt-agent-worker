import { Course } from "./course";

export interface AgentResponse {
    chat: string;
    courseList: Course[]; 
}