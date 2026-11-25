"""Start the existing LangGraph as a Foundry Responses agent."""

from langchain_azure_ai.agents.hosting.run import main  # type: ignore[import-untyped]

if __name__ == "__main__":
    main(["--protocol", "responses"])
