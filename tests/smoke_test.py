import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.pipeline import RAGPipeline

def main():
    p = RAGPipeline()
    print("[1/3] Loading pipeline and collection...")
    stats = p.vector_store.get_stats()
    print(f"Indexed collection contains: {stats.get('total_notes')} notes, {stats.get('total_chunks')} chunks.")

    print("\n[2/3] Testing Groq provider...")
    if p.groq_client.is_configured():
        try:
            res = p.query("What is our database and vector storage strategy?", provider="groq")
            print("Groq Status: SUCCESS")
            print(f"Provider: {res.provider} ({res.model})")
            print("Answer preview:", res.answer[:200].replace('\n', ' '), "...")
            print(f"Citations count: {len(res.citations)}")
            for c in res.citations[:2]:
                print(f" - [{c.source_file}: {c.heading}] ({c.similarity_score * 100:.1f}%)")
        except Exception as e:
            print("Groq Status: FAILED -", e)
    else:
        print("Groq Status: KEY NOT DETECTED")

    print("\n[3/3] Testing Gemini provider...")
    if p.gemini_client.is_configured():
        try:
            res = p.query("What are our core architectural pillars?", provider="gemini")
            print("Gemini Status: SUCCESS")
            print(f"Provider: {res.provider} ({res.model})")
            print("Answer preview:", res.answer[:200].replace('\n', ' '), "...")
            print(f"Citations count: {len(res.citations)}")
            for c in res.citations[:2]:
                print(f" - [{c.source_file}: {c.heading}] ({c.similarity_score * 100:.1f}%)")
        except Exception as e:
            print("Gemini Status: FAILED -", e)
    else:
        print("Gemini Status: KEY NOT DETECTED")

    print("\n[4/4] Testing LLMRouter (Auto Fallback Chain)...")
    try:
        res_router = p.query("What architectural patterns are used for scalability?", provider="auto")
        print("LLMRouter Status: SUCCESS")
        print(f"Provider: {res_router.provider} ({res_router.model})")
        print("Answer preview:", res_router.answer[:200].replace('\n', ' '), "...")
        print(f"Citations count: {len(res_router.citations)}")
    except Exception as e:
        print("LLMRouter Status: FAILED -", e)

if __name__ == "__main__":
    main()
