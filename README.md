# Task 1: Text Classification Microservice
## Objective
Create a Python microservice that classifies text input into predefined categories using an AI model.

## Instructions
1. Project Description: Develop a Python microservice that takes a text input, sends it to an AI model for classification, and returns the category in JSON format.

2. Requirements:
- Use any Python web framework, such as Flask or FastAPI.
- Create a function to call the AI model (e.g., using the OpenAI API or another available AI API).
- Develop an endpoint that receives text input and returns the classification result in JSON format.
- The categories for classification can be predefined (e.g., "sports", "politics", "technology"). 
3. Bonus (Optional):
- Include appropriate exception handling and response codes.
- Provide a README.md with setup and run instructions, along with an explanation of the implemented logic.
- Implement unit tests for the service.

## Evaluation Criteria
- Functionality: The microservice works as described and handles errors gracefully.
- Code Quality: Clean, readable, and well-organized code.
- Bonus: Clear instructions and explanation in README.md, presence of unit tests.

# Task 2: Dockerization

## Objective
Containerize the project from Task 1.

## Instructions
1. Create a Dockerfile to containerize your project.
2. Create/update a README.md that explains how to build and run the Docker container.
3. Focus on creating a basic but correct Dockerfile.

## Evaluation Criteria
- Correct setup of the Dockerfile.
- Ability to successfully build and run the container.
- Clarity of instructions in README.md.

# Task 3: Using Git and GitHub
## Objective
Demonstrate basic Git and GitHub skills.

## Instructions
1. Create a public repository on GitHub.com.
2. For your work from Tasks 1 and 2, create a new branch named <your-surname>-python-task, upload all files, and push your solution to this branch on GitHub.
3. Add josef.polach@granton.cz as a collaborator to this public project.
4. Create a new pull request on GitHub to prepare your project for merging your changes into the main branch. (Just create the request, do not merge the changes yet)

## Evaluation Criteria
- Correct use of Git for branching and committing changes.
- Successful creation of a pull request.
- Understanding of version control concepts.

# Requirements:

1. Prepare an env file with required API keys according to model used (so far only OpenAI was implemented, but left room for expansion)
- OpenAI: env var `OPENAI_API_KEY`

# Build:

`docker build -t [build name] [path to project]`

# Run:

`docker run --env-file [env file path] -p 5000:5000 [build name]`