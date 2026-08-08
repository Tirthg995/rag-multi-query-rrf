# ── src/eval_data.py ──

EVAL_QUESTIONS = [
    {
        "question": "What is task decomposition for LLM agents?",
        "ground_truth": "Task decomposition is the process of breaking a complex task into smaller, manageable subgoals or steps. It can be done via simple LLM prompting (e.g. asking for steps), task-specific instructions, or human input, enabling the agent to plan ahead."
    },
    {
        "question": "How does self-reflection improve an LLM agent's performance?",
        "ground_truth": "Self-reflection allows an agent to look back on past actions and outcomes, identify mistakes or inefficiencies, and refine its future plans. Techniques like ReAct and Reflexion combine reasoning traces with acting, and use reflection to iteratively improve decision-making over multiple rounds."
    },
    {
        "question": "What types of memory does an LLM agent use?",
        "ground_truth": "LLM agents typically use short-term memory (the in-context learning within the prompt window) and long-term memory (an external vector store that allows retention and retrieval of information over extended periods, accessed via fast approximate nearest neighbor search)."
    },
    {
        "question": "How do LLM agents use external tools?",
        "ground_truth": "LLM agents call external APIs or tools to access capabilities beyond their built-in knowledge, such as real-time information, code execution, or proprietary data sources. Approaches like MRKL, Toolformer, and HuggingGPT teach or prompt models to select and invoke appropriate external tools for a given task."
    },
    {
        "question": "What is the role of a vector store in agent memory?",
        "ground_truth": "A vector store enables long-term memory by storing embedded representations of information, which the agent can retrieve later using approximate nearest neighbor (ANN) search algorithms, allowing fast access to relevant past information beyond the LLM's limited context window."
    },
    {
        "question": "What are the main limitations of current LLM-powered autonomous agents?",
        "ground_truth": "Key limitations include finite context length restricting how much information can be considered at once, difficulty with long-term planning and adjusting plans given unexpected errors, and unreliability of natural language as an interface since the LLM may make formatting or reasoning mistakes."
    },
]
HARD_EVAL_QUESTIONS = [
    {
        "question": "How do planning, memory, and tool use work together in an LLM agent's architecture?",
        "ground_truth": "Planning breaks a task into subgoals and enables reflection to refine future actions; memory (short-term via context, long-term via a vector store) lets the agent retain and retrieve information beyond its context window; tool use extends the agent's capabilities by calling external APIs. Together, these components let an agent decompose a complex task, remember relevant past information, and act on it using external resources."
    },
    {
        "question": "What approaches help an LLM agent recover from or learn from its own mistakes?",
        "ground_truth": "Self-reflection mechanisms like ReAct and Reflexion let an agent review past actions and outcomes, identify errors, and incorporate that feedback into future decisions, enabling iterative improvement across multiple attempts at a task."
    },
    {
        "question": "Why is long-term memory important for an LLM agent, and how is it typically implemented?",
        "ground_truth": "Long-term memory allows an agent to retain and recall information beyond the limits of its context window over extended periods. It's typically implemented using an external vector store paired with approximate nearest neighbor (ANN) search, allowing fast retrieval of relevant past information."
    },
    {
        "question": "Compare how task decomposition and self-reflection each contribute to an agent's ability to handle complex tasks.",
        "ground_truth": "Task decomposition breaks a complex task into smaller, manageable subgoals so the agent can plan a path to completion, while self-reflection lets the agent evaluate the outcomes of its actions and refine its approach over time. Decomposition provides the initial structure for tackling a task; reflection improves execution quality iteratively."
    },
    {
        "question": "What combination of limitations makes long-horizon planning difficult for current LLM agents?",
        "ground_truth": "Long-horizon planning is difficult due to the LLM's finite context length limiting how much information can be considered at once, difficulty adjusting plans when unexpected errors occur, and the unreliability of natural language as an interface, which can introduce formatting or reasoning mistakes that compound over many steps."
    },
]

# combine sets of questions
EVAL_QUESTIONS = EVAL_QUESTIONS + HARD_EVAL_QUESTIONS