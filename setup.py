from setuptools import setup, find_packages

setup(
    name="agentic-trading-system",
    version="0.0.2",
    author="Sowmya AM",
    author_email="sowmya.anekonda@gmail.com",
    packages=find_packages(),
    python_requires=['langchain', 'lancedb', 'langgraph', 'tavily-python', 'polygon']
)