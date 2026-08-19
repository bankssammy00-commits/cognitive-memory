from app.memory.extractor import MemoryExtractor


extractor = MemoryExtractor()


text = """
I stopped using Supplier X because they kept delivering late.
Their average delivery was around 21 days, while I need suppliers
to deliver within 14 days.
"""

memories = extractor.extract(text)

print("\nEXTRACTED MEMORIES:\n")

for memory in memories:
    print(memory.model_dump_json(indent=2))