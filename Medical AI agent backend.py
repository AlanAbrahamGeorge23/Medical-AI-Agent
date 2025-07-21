import os
import getpass
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
#API key secure
os.environ['GOOGLE_API_KEY'] = getpass.getpass("Enter the Gemini API Key")
#initialize gemini flash
llm = ChatGoogleGenerativeAI(model = "models/gemini-1.5-flash-latest", temperature =0.2)
#node to ask for symptom
def get_symptom(state: dict) ->dict:
  symptom = input("Welcome to Alan Medical Assistant, Please enter your symptom ")
  state["symptom"] = symptom
  return state
#node to classify symptom
def classify_symptom(state:dict) -> dict:
  symptom = state.get("symptom", "No symptom provided") # Get the symptom from the state
  prompt = (
      "You are a helpful Medical Assistant, Classify the following symptom into one of the categories \n"
      "-General\n -Emergency \n -mental health \n"
      "Respond only with one word : General, Emergency Or Mental Health\n"
      f"Symptom: {state['symptom']}\n" # Include the symptom in the prompt
      "#Example : input : I have fever, Output : General"
  )
  response = llm.invoke([HumanMessage(content=prompt)])
  category = response.content.strip()
  print(f"LLM classifies the symptom as : {category}")  #debug
  state["category"] = category # Add the classified category to the state
  return state
#route to handle the classified symptom
def symptom_router(state: dict) -> dict:
  cat = state["category"].lower()
  if "general" in cat :
    return "general"
  elif "emergency" in cat:
    return "emergency"
  elif "mental" in cat:
    return "mental_health"
  else:
    return "general"
# category specific response
def general_node(state: dict) -> dict:
  state["answer"] = f" '{state ['symptom']}' : seems general : directing you to general ward for consulting a doctor"
  return state

def emergency_node(state:dict) -> dict:
  state["answer"] =f" '{state ['symptom']}' : It is a Medical Emergency : seeking immediate help"
  return state
def mental_health_node(state:dict) -> dict:
  state["answer"] = f" '{state ['symptom']}' : seems like a medical health issue: talk to our counsellor"
  return state
# build langgraph
from typing import TypedDict

class AgentState(TypedDict):
    symptom: str
    category: str
    answer: str

builder = StateGraph(AgentState)  # Pass the schema to StateGraph
#define the nodes
builder.set_entry_point("get_symptom")
builder.add_node("get_symptom", get_symptom)
builder.add_node("classify",classify_symptom)
builder.add_node("general",general_node)
builder.add_node("emergency",emergency_node)
builder.add_node("mental_health",mental_health_node)
builder.add_edge("get_symptom","classify")
builder.add_conditional_edges("classify", symptom_router, {
    "general": "general",
    "emergency": "emergency",
    "mental_health": "mental_health"
})
builder.add_edge("general", END)
builder.add_edge("emergency", END)
builder.add_edge("mental_health", END)
# Invoke graph
graph = builder.compile()
final_state = graph.invoke({})
print("final output \n")
print(final_state["answer"])


