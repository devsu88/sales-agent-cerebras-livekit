import os
from pathlib import Path
from urllib.request import urlretrieve

from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, function_tool
from livekit.plugins import cartesia, openai, silero

load_dotenv(override=True)

CONTEXT_DIR = Path("./content/context")
PRODUCTS_URL = (
    "https://gist.githubusercontent.com/ShayneP/f373c26c5166d90446f2bc08baf9bf46/raw/products.json"
)


def ensure_context_files() -> None:
    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    products_path = CONTEXT_DIR / "products.json"
    if not products_path.exists():
        urlretrieve(PRODUCTS_URL, products_path)


def load_context() -> str:
    context_dir = CONTEXT_DIR
    context_dir.mkdir(parents=True, exist_ok=True)

    all_content = ""
    for file_path in context_dir.glob("*"):
        if file_path.is_file():
            try:
                content = file_path.read_text(encoding="utf-8")
                all_content += f"\n=== {file_path.name} ===\n{content}\n"
            except Exception:
                pass

    return all_content.strip() or "No files found"

def build_voice_stack(voice: str = None):
    llm = openai.LLM.with_cerebras(model="llama3.1-8b")
    stt = cartesia.STT()
    tts = cartesia.TTS(voice=voice) if voice else cartesia.TTS()
    vad = silero.VAD.load()
    return llm, stt, tts, vad


class SalesAgent(Agent):
    def __init__(self):
        sales_context = load_context()
        llm, stt, tts, vad = build_voice_stack()
        instructions = f"""
        You are a sales agent communicating by voice. All text that you return
        will be spoken aloud, so don't use things like bullets, slashes, or any
        other non-pronouncable punctuation.

        You have access to the following company information:

        {sales_context}

        CRITICAL RULES:
        - ONLY use information from the context above
        - If asked about something not in the context, say "I don't have that information"
        - DO NOT make up prices, features, or any other details
        - Quote directly from the context when possible
        - Be a sales agent but only use the provided information
        """

        super().__init__(instructions=instructions, stt=stt, llm=llm, tts=tts, vad=vad)

    async def on_enter(self):
        print("Current Agent: Sales Agent")
        self.session.generate_reply(
            user_input="Give a short, 1 sentence greeting. Offer to answer any questions."
        )

    @function_tool
    async def switch_to_tech_support(self):
        await self.session.generate_reply(
            user_input="Confirm you are transferring to technical support"
        )
        return TechnicalAgent()

    @function_tool
    async def switch_to_pricing(self):
        await self.session.generate_reply(
            user_input="Confirm you are transferring to a pricing specialist"
        )
        return PricingAgent()


class TechnicalAgent(Agent):
    def __init__(self):
        sales_context = load_context()
        llm, stt, tts, vad = build_voice_stack(voice="bf0a246a-8642-498a-9950-80c35e9276b5")
        instructions = f"""
        You are a technical specialist communicating by voice. All text that you return
        will be spoken aloud, so don't use things like bullets, slashes, or any
        other non-pronouncable punctuation.

        You specialize in technical details, specifications, and implementation questions.
        Focus on technical accuracy and depth.

        You have access to the following company information:

        {sales_context}

        CRITICAL RULES:
        - ONLY use information from the context above
        - Focus on technical specifications and features
        - Explain technical concepts clearly for non-technical users
        - DO NOT make up technical details

        You can transfer to other specialists:
        - Use switch_to_sales() to return to general sales
        - Use switch_to_pricing() for pricing questions
        """
        super().__init__(instructions=instructions, stt=stt, llm=llm, tts=tts, vad=vad)

    async def on_enter(self):
        print("Current Agent: Technical Specialist")
        await self.session.say(
            "Hi, I'm the technical specialist. I can help you with technical questions."
        )

    @function_tool
    async def switch_to_sales(self):
        await self.session.generate_reply(
            user_input="Confirm you are transferring to the sales team"
        )
        return SalesAgent()

    @function_tool
    async def switch_to_pricing(self):
        await self.session.generate_reply(
            user_input="Confirm you are transferring to a pricing specialist"
        )
        return PricingAgent()


class PricingAgent(Agent):
    def __init__(self):
        sales_context = load_context()
        llm, stt, tts, vad = build_voice_stack(voice="4df027cb-2920-4a1f-8c34-f21529d5c3fe")
        instructions = f"""
        You are a pricing specialist communicating by voice. All text that you return
        will be spoken aloud, so don't use things like bullets, slashes, or any
        other non-pronouncable punctuation.

        You specialize in pricing, budgets, discounts, and financial aspects.
        Help customers find the best value for their needs.

        You have access to the following company information:

        {sales_context}

        CRITICAL RULES:
        - ONLY use pricing information from the context above
        - Focus on value proposition and ROI
        - Help customers understand pricing tiers and options
        - DO NOT make up prices or discounts

        You can transfer to other specialists:
        - Use switch_to_sales() to return to general sales
        - Use switch_to_technical() for technical questions
        """
        super().__init__(instructions=instructions, stt=stt, llm=llm, tts=tts, vad=vad)

    async def on_enter(self):
        print("Current Agent: Pricing Agent")
        await self.session.say(
            "Hello, I'm the pricing specialist. I can help with pricing options."
        )

    @function_tool
    async def switch_to_sales(self):
        await self.session.generate_reply(
            user_input="Confirm you are transferring to the sales team"
        )
        return SalesAgent()

    @function_tool
    async def switch_to_technical(self):
        await self.session.generate_reply(
            user_input="Confirm you are transferring to technical support"
        )
        return TechnicalAgent()


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    session = AgentSession()
    await session.start(room=ctx.room, agent=SalesAgent())


def main() -> None:
    ensure_context_files()
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))


if __name__ == "__main__":
    main()
