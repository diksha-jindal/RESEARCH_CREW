# Retrieval-Augmented Generation
- Retrieval-augmented generation supplies a language model with relevant documents at question time instead of relying only on its training data.
- A typical pipeline splits documents into chunks, indexes them, retrieves the most relevant chunks, and passes them into the prompt.
- Grounding answers in retrieved text makes it possible to attach source citations to each claim.
- Retrieval quality limits answer quality, because the model cannot cite what the retriever fails to find.
- RAG helps keep answers current because the document store can be updated without retraining the model.
