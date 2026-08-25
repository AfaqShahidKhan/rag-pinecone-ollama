# Test Landing Zone

Drop test files here — kept separate from `data/landing_zone/` so testing
document parsing never mixes with real ingestion data.

## How to run

    python main.py build-corpus data/test_landing_zone --config config/user_testing.yml

No Ollama, no Pinecone/Chroma/Qdrant connection required — `build-corpus`
only runs load -> image extraction -> pre-processing -> corpus writing. It
stops before chunking/embedding/upsert entirely.

Output lands in `data/corpus/<filename>/page_NNN.md` (or a separate
location if you set `corpus.output_dir` in a config profile — recommended,
so test runs don't overwrite your real corpus output; see
`config/user_template.yml`).

## What to check per file type

| File you add                                      | What to verify in the resulting corpus .md file(s)                                                                                         |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| PDF with at least one table                       | `table_count` > 0 in front matter; a real Markdown table (`\| col \| col \|`) appears in the body, not flattened text                      |
| PDF with an embedded image                        | `image_count` > 0 in front matter; check `data/images/<name>/img_*.png` exists                                                             |
| PDF that's a scanned/no-text page                 | Confirm it's skipped with a clear warning in `logs/rag.log`, not a silent gap                                                              |
| DOCX with a table                                 | Same table checks as PDF                                                                                                                   |
| PPTX with a table + speaker notes                 | `table_count` > 0; body contains a `--- Speaker Notes ---` section                                                                         |
| Legacy `.ppt` (PowerPoint 97-2003)                | Requires LibreOffice installed; check `logs/rag.log` for the conversion log line, and confirm output matches a `.pptx` of the same content |
| JSON file                                         | Content appears correctly structured in the corpus, `file_type: json`                                                                      |
| HTML file                                         | Content appears correctly structured in the corpus, `file_type: html`                                                                      |
| A file containing sample PII (email, phone, etc.) | `pii_redacted: true` in front matter; the actual PII value does NOT appear in the body — a placeholder does                                |

## Tip: isolate test output from real data

Create `config/user_testing.yml` (copy `config/user_template.yml`) with:

```yaml
corpus:
  output_dir: ./data/test_corpus
image_extraction:
  output_dir: ./data/test_images
```

Then test runs never touch your real `data/corpus/` or `data/images/`.
