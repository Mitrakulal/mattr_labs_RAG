import re
import ollama
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter



ENCODING = tiktoken.get_encoding("cl100k_base")

CHUNK_SIZE_TOKENS=400
CHUNK_OVERLAP_TOKENS=50

def token_length(text: str) -> int:
    """Return the number of tokens in `text`, used as the splitter's length function."""
    return len(ENCODING.encode(text))


def load_source_text(path:str)->str:
    with open(path,"r",encoding="utf-8") as f:
        return f.read();
    

def split_into_sections(full_text:str)->list[dict]:
    """ Split the raw file into its labeled sections
    '#===XXX=====#' kind of heading """
    
    sections=[]
    current_section="Unlabeled"
    current_lines=[]
    
    section_pattern = re.compile(r"^#\s*=+\s*(.*?)\s*=+\s*$")
    
    for line in full_text.splitlines():
        
        stripped=line.strip()
        match = section_pattern.match(stripped)
        
        if match:
            if current_lines:
                sections.append(
                    {"section":current_section,
                     "text":"\n".join(current_lines).strip()
                     })
                current_section=stripped.strip("# =").strip()
                current_lines=[]
        else:
            current_lines.append(line)
            
    if current_lines:
        sections.append({
            "section":current_section,
            "text":"\n".join(current_lines).strip()
        })
    
    return [s for s in sections if s["text"]]
                

def chunk_sections(sections:list[dict])->list[dict]:
    """Apply recusrive , token based chuckning to each sections independently"""
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        length_function=token_length,
        separators=["\n\n","\n",". "," ", ""],
        
    )
    all_chunks=[]
    
    for section in sections:
        pieces=splitter.split_text(section["text"])
        for i , piece in enumerate(pieces):
            all_chunks.append({
                "text":piece,
                "section":section["section"],
                "chunk_index":i,
            })
            
    return all_chunks