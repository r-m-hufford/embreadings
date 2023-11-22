# Embreadings 🍞

## What is this?

This is a simple python script that creates embeddings in a vector database (Astra by Datastax) from text files.
It uses...

- AstraDB
- Langchain
- OpenAI
- Cassandra

## What does it do?

It fetches a bunch of Onion articles, shoves them in a database and then uses the OpenAI API (via LangChain) to create embeddings for each article. After that, you can QA Onion articles to get very serious answers to very serious questions. ai is the future.
