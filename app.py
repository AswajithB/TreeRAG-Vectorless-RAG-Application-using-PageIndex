import streamlit as st
import os
import time
import json
import tempfile
import base64
from pageindex import PageIndexClient
from groq import Groq
from dotenv import load_dotenv

st.set_page_config(page_title="TreeRAG - Vectorless RAG", page_icon="🌲", layout="centered")

# Load environment variables silently
load_dotenv()

# We strictly pull the keys from the backend env variables
pageindex_key = os.getenv("PAGEINDEX_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

if not pageindex_key or not groq_key:
    st.error("⚠️ API Keys are missing! Please set PAGEINDEX_API_KEY and GROQ_API_KEY in your .env file.")
    st.stop()

# Initialize API clients
pi_client = PageIndexClient(api_key=pageindex_key)
groq_client = Groq(api_key=groq_key)

with st.sidebar:
    st.markdown("## 🏆 Powered by [PageIndex](https://dash.pageindex.ai/)")
    st.markdown("---")
    st.info("💡 **Vectorless RAG**\n\nPageIndex reads documents perfectly by mapping their native layout. No chunking, no vectors, no lost context.")
    st.markdown("🔗 [Get your API Key](https://dash.pageindex.ai)")
   

st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap; padding-bottom: 20px;">
        <h1 style="margin: 0; padding: 0;">🌲 TreeRAG: Chat with PDFs</h1>
        <span style="background-color: #2e7d32; color: white; padding: 4px 14px; border-radius: 20px; font-weight: 700; font-size: 14px; letter-spacing: 0.5px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">🔥 VECTORLESS RAG</span>
    </div>
    """, 
    unsafe_allow_html=True
)
st.markdown("Upload a **structured PDF** (e.g., Research Papers, Reports, Textbooks). The AI reads its native hierarchy to find exact answers without chunking or vector DBs.")

st.info("**💡 Ideal Use Cases & Constraints:** TreeRAG is highly optimized for documents with clear section headings, chapters, and a logical structure. It maps the document's 'Tree'. Completely unstructured text or purely visual/scanned PDFs will not perform as accurately.")

# Initialize session states
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "tree" not in st.session_state:
    st.session_state.tree = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm here to help. You can just say 'Hi', or upload a PDF to start asking document-specific questions!"}
    ]

# ── 1. PDF File Uploader ──
uploaded_file = st.file_uploader("Upload your document here", type=["pdf"])

if uploaded_file:
    # Check if this is a new file that we haven't processed yet
    if st.session_state.uploaded_filename != uploaded_file.name:
        st.session_state.doc_id = None
        st.session_state.tree = None
        st.session_state.uploaded_filename = uploaded_file.name
        
    # If the file hasn't been uploaded to PageIndex, process it now
    if st.session_state.doc_id is None:
        with st.spinner(f"📤 Uploading '{uploaded_file.name}' to PageIndex..."):
            # Save uploaded file temporarily to pass path to SDK
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
                
            try:
                # Submit to PageIndex
                upload_result = pi_client.submit_document(tmp_path)
                doc_id = upload_result["doc_id"]
                st.session_state.doc_id = doc_id
                
                # Poll until the document is structurally mapped (Tree is ready)
                status_placeholder = st.empty()
                while True:
                    status_result = pi_client.get_document(doc_id)
                    status = status_result.get("status")
                    
                    if status == "completed":
                        status_placeholder.success("✅ AI mapping completed!")
                        time.sleep(1) # Let the user see success message before hiding
                        status_placeholder.empty() 
                        break
                    elif status == "failed":
                        status_placeholder.error("❌ Document processing failed on server side.")
                        os.remove(tmp_path)
                        st.stop()
                    else:
                        status_placeholder.info(f"⏳ Building hierarchical tree structure... (Status: {status})")
                    time.sleep(4)
                
                # Fetch final structural tree
                with st.spinner("🌲 Fetching document layout..."):
                    tree_result = pi_client.get_tree(doc_id, node_summary=True)
                    st.session_state.tree = tree_result.get("result", [])
                    
            except Exception as e:
                st.error(f"Error during upload process: {str(e)}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

# Display success text if file is ready
if st.session_state.tree:
    st.success(f"📄 '{st.session_state.uploaded_filename}' is ready! Ask your questions below.")

    with st.expander("🌳 View Internal Tree Structure", expanded=False):
        st.markdown("This JSON represents the hierarchical structure that PageIndex parsed, mapping sections, pages, and summaries dynamically.")
        with st.container(height=400):
            st.json(st.session_state.tree)

st.markdown("---")

# ── 2. Chat Interface ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Say 'hello' or ask a direct question about your document..."):
    # Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        # We need an intelligent router: Is this a normal chat or a PDF query?
        classifier_prompt = f"""You are a smart routing assistant. Classify the user's message as 'GENERAL_CHAT' if it is a greeting (e.g. 'hi', 'hello', 'thanks', 'how are you'). Classify it as 'DOCUMENT_QUERY' if it is a specific question, or asks for facts/summary.
        User message: "{prompt}"
        Reply ONLY with 'GENERAL_CHAT' or 'DOCUMENT_QUERY'. Do not add any other text."""
        
        try:
            clf_response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": classifier_prompt}],
                temperature=0.1
            )
            intent = clf_response.choices[0].message.content.strip()
            # Failsafe clean
            if "GENERAL" in intent: intent = "GENERAL_CHAT"
            elif "DOCUMENT" in intent: intent = "DOCUMENT_QUERY"
        except:
            intent = "DOCUMENT_QUERY"  # default
            
        # Branch 1: General Chat Handling
        if intent == "GENERAL_CHAT":
            chat_context = "You are a helpful and polite AI. Provide a friendly conversational reply. Do not offer random facts."
            if not st.session_state.tree:
                chat_context += " Remind them to upload a PDF if they want to ask specific questions about a document."
                
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": chat_context}, {"role": "user", "content": prompt}],
                temperature=0.7,
                stream=True
            )
            
            def stream_generator(resp):
                for chunk in resp:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
                        
            answer = st.write_stream(stream_generator(response))
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        # Branch 2: Document Query Handling but no PDF provided
        elif intent == "DOCUMENT_QUERY" and not st.session_state.tree:
            answer = "I'd love to answer that, but I don't have a document yet! 📄 Please upload a PDF file using the uploader above first."
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        # Branch 3: Actively Search the PDF
        elif intent == "DOCUMENT_QUERY" and st.session_state.tree:
            with st.spinner("🧠 Reasoning over Document Table of Contents..."):
                def compress(nodes):
                    out = []
                    for n in nodes:
                        entry = {
                            "node_id": n["node_id"],
                            "title":   n["title"],
                            "page":    n.get("page_index", "?"),
                            "summary": n.get("text", "")[:150],
                        }
                        if n.get("nodes"):
                            entry["children"] = compress(n["nodes"])
                        out.append(entry)
                    return out

                compressed_tree = compress(st.session_state.tree)

                search_prompt = f"""You are analyzing a document's tree structure.
Find all node IDs that most likely contain the answer to the query.
Query: {prompt}
Document Tree: {json.dumps(compressed_tree, indent=2)}
Reply ONLY in exact JSON format: {{"thinking": "your step-by-step reasoning...", "node_list": ["id1", "id2"]}}"""

                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": search_prompt}],
                    response_format={"type": "json_object"}
                )
                
                search_result = json.loads(response.choices[0].message.content)
                node_ids = search_result.get("node_list", [])
                thinking_logic = search_result.get("thinking", "No reasoning provided.")
                
            if not node_ids:
                answer = "I couldn't identify any relevant sections in the document structure to accurately answer this question."
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                with st.expander("💭 AI Routing Logic (Navigating Tree)", expanded=False):
                    st.info(thinking_logic)
                    st.caption(f"Targeted nodes: {node_ids}")

                with st.spinner("📄 Retrieving targeted sections..."):
                    def find_nodes(tree, target_ids):
                        found = []
                        for node in tree:
                            if node["node_id"] in target_ids:
                                found.append(node)
                            if node.get("nodes"):
                                found.extend(find_nodes(node["nodes"], target_ids))
                        return found
                    
                    nodes = find_nodes(st.session_state.tree, node_ids)
                    
                    context_parts = []
                    for node in nodes:
                        context_parts.append(
                            f"[Section: '{node['title']}' | Page {node.get('page_index', '?')}]\n"
                            f"{node.get('text', 'Content not available.')}"
                        )
                    context = "\n\n---\n\n".join(context_parts)
                    
                answer_prompt = f"""Answer the question using ONLY the provided context sections.
For every claim you make, explicitly cite the section title and page number in parentheses at the end of the sentence. Do not make up answers not found in the text.
Question: {prompt}

Context: {context}"""

                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": answer_prompt}],
                    temperature=0.2,
                    stream=True
                )
                
                def stream_generator(resp):
                    for chunk in resp:
                        if chunk.choices[0].delta.content is not None:
                            yield chunk.choices[0].delta.content
                            
                answer = st.write_stream(stream_generator(response))
                
                history_content = f"**💭 Routing Logic:** {thinking_logic}\n\n---\n\n**🤖 Answer:** {answer}"
                st.session_state.messages.append({"role": "assistant", "content": history_content})
