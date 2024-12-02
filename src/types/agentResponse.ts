import { Course } from "./course";

export interface AgentResponse {
    answer: string;
    courseList: Course[]; 
}