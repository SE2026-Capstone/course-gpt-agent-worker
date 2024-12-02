export interface Course {
    courseCode: string;
    courseName: string;
    courseDescription: string;
    relevanceScore: number; // vector match score
}