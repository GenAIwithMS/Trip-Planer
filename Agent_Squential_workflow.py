from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate


load_dotenv()

model = ChatGroq(model="gemma2-9b-it")

class Statetravel(TypedDict):

    destination: str
    outline: str = None
    itinerary: str = None
    budget : str = None



def create_outline(state: Statetravel) -> Statetravel:

    destination = state["destination"]

    prompt = f"""User travel request: {destination}

✨ Task: Based on the above request, create a clear travel outline.  
- Start with a short introduction about the destination.  
- Summarize the best highlights/attractions.  
- Suggest a high-level trip structure (e.g., Arrival, Sightseeing, Adventure, Relaxation).  
- Keep it concise and organized — not a detailed itinerary yet.  
"""


    outline = model.invoke(prompt).content
    return {"outline":outline}

def detailed_itinerary(state: Statetravel) -> Statetravel:

    destination = state["destination"]

    prompt = f"""User travel request: {destination}

✨ Task: Create a detailed **day-by-day itinerary** for the trip.  
- Use the number of days mentioned in the request.  
- Break down each day into Morning, Afternoon, and Evening activities.  
- Recommend sightseeing, food, and cultural experiences.  
- Mix famous attractions with hidden gems.  
- Keep it realistic and easy to follow.  

Format Example:  
Day 1: ...  
Day 2: ...  
"""

    itinerary = model.invoke(prompt).content
    
    return {"itinerary": itinerary}

def total_budget(state: Statetravel) -> Statetravel:
    
    destination = state["destination"]
    
    prompt = f"""User travel request: {destination}

✨ Task: Estimate the **total budget** for this trip.  
- Consider the destination, trip duration, and budget preference (Luxury, Normal, or Low Cost) mentioned in the request.  
- Provide a cost breakdown: Flights, Hotels, Food, Local Transport, Activities.  
- Show an approximate price range in the preferd curruncy in request.  
- Mention inclusions/exclusions.  
- Add tips: how to save (for low cost) or upgrade (for luxury).  
"""

    budget = model.invoke(prompt).content
    
    

    return {"budget": budget}

graph = StateGraph(Statetravel)

graph.add_node("create_outline", create_outline)
graph.add_node("detailed_itinerary", detailed_itinerary)
graph.add_node("total_budget",total_budget)

graph.add_edge(START,"create_outline")
graph.add_edge("create_outline","detailed_itinerary")
graph.add_edge("detailed_itinerary","total_budget")
graph.add_edge("total_budget",END)

result = graph.compile()


#==================================> frontend <============================================= #

st.header("GlobePath")

destination = st.text_input(label="Enter Your Destination below")
days = st.number_input("Insert days of trip")
budget = st.selectbox("Select your budget", ("Luxury","Normal","Low Cost"))
curruncy = st.selectbox("select your prefferd curruncy", ("PKR","USD","EUR"))

prompt = PromptTemplate.from_template("I want to visit {destination} for {days} days with a budget of {budget} in {curruncy}")

final_prompt = prompt.format(destination=destination, days=days, budget=budget, curruncy=curruncy)

button = st.button("Submit")

if button and final_prompt:
    final_state = result.invoke({"destination": final_prompt})

    # Show nicely formatted sections
    st.subheader(f"✈️ Destination: {final_state['destination']}")

    with st.expander("📌 Outline", expanded=True):
        st.markdown(final_state["outline"])

    with st.expander("🗺️ Detailed Itinerary", expanded=False):
        st.markdown(final_state["itinerary"])

    with st.expander("💰 Estimated Budget", expanded=False):
        st.markdown(final_state["budget"])
