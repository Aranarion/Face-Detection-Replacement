# Face-Detection-Replacement

- Implemented a TCP client for the \texttt{uqfacedetect} server to submit face‑detection or face‑replacement requests—sending binary images, receiving processed output with detected faces outlined or faces substituted, and persisting results to disk or stdout.
- Engineered the client–server protocol layer to establish localhost TCP connections, frame little‑endian requests for detect‑only or replace operations, stream large image payloads bidirectionally in fixed‑size chunks, parse responses (image vs error), handle communication failures, and ensure clean resource and memory management.
