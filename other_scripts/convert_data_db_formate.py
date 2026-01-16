import csv
import json

INPUT_CSV = "other_scripts/interview_output.csv"
OUTPUT_JSON = "other_scripts/output2.json"

final_output = []

with open(INPUT_CSV, newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        # Parse emotions JSON (string → dict)
        emotions_dict = json.loads(row["emotions_json"])

        # Convert dict → sorted list
        hume_ai_expression = sorted(
            [
                {
                    "name": emotion,
                    "score": float(score)
                }
                for emotion, score in emotions_dict.items()
            ],
            key=lambda x: x["score"],
            reverse=True
        )

        result = {
            "imageUrl": row["image_url"],
            "cropImageUrl": row["image_url"],
            "jobIdFromOurModel": None,
            "modelExpression": None,
            "humeAIExpression": hume_ai_expression,
            "trained": True
        }

        final_output.append(result)

# Write final JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=2)

print(f"✅ JSON file created: {OUTPUT_JSON}")
