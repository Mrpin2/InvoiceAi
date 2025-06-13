        with st.spinner("🧠 Extracting data using AI..."):
            try:
                if model_choice == "Gemini" and gemini_api_key:
                    client = genai.Client(api_key=gemini_api_key)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(pdf_data)
                        temp_path = tmp.name

                    uploaded = client.files.upload(file=temp_path, config={"display_name": file_name})
                    os.unlink(temp_path)

                    prompt = (
                        "Extract invoice data as structured JSON. Use DD/MM/YYYY format. "
                        "Leave missing fields empty or null. Do not hallucinate. Focus on Indian invoice fields."
                    )
                    response = client.models.generate_content(
                        model="gemini-1.5-flash-latest",
                        contents=[prompt, uploaded],
                        config={
                            "response_mime_type": "application/json",
                            "response_schema": Invoice
                        }
                    )
                    client.files.delete(name=uploaded.name)
                    data = response.parsed

                    narration = (
                        f"Invoice {data.invoice_number} dated {data.date} issued by {data.seller_name} "
                        f"(GSTIN: {data.gstin}) to {data.buyer_name} "
                        f"(GSTIN: {data.buyer_gstin or '-'}) for ₹{data.total_gross_worth:.2f}. "
                        f"Taxes: CGST ₹{data.cgst or 0.0}, SGST ₹{data.sgst or 0.0}, IGST ₹{data.igst or 0.0}. "
                        f"Ledger: {data.expense_ledger or '-'}, POS: {data.place_of_supply or '-'}, "
                        f"TDS: {data.tds or '-'}."
                    )

                    result_row = [
                        file_name,
                        data.seller_name,
                        data.invoice_number,
                        data.date,
                        data.expense_ledger or "-",
                        "CGST+SGST" if data.cgst and data.sgst else ("IGST" if data.igst else "NA"),
                        "-",  # Tax Rate not explicitly given
                        data.total_gross_worth - sum([data.cgst or 0, data.sgst or 0, data.igst or 0]),
                        data.cgst or 0.0,
                        data.sgst or 0.0,
                        data.igst or 0.0,
                        data.total_gross_worth,
                        narration,
                        "Yes" if data.expense_ledger and "travel" not in data.expense_ledger.lower() else "No",
                        "Yes" if data.tds and data.tds.lower().startswith("yes") else "No",
                        data.tds.split()[-1] if data.tds and "%" in data.tds else "0"
                    ]

                elif model_choice == "ChatGPT" and openai_api_key:
                    # Keep your old ChatGPT image-based code if needed here
                    st.warning("ChatGPT not updated for schema-based parsing. Use Gemini instead.")
                    result_row = [file_name] + ["NOT AN INVOICE"] + ["-"] * (len(columns) - 2)

                else:
                    raise Exception("❌ No valid API key provided.")

                st.session_state["processed_results"][file_name] = result_row

            except Exception as e:
                st.error(f"❌ Error processing {file_name}: {e}")
                st.text_area(f"Raw Output ({file_name})", traceback.format_exc())
                st.session_state["processed_results"][file_name] = [file_name] + ["NOT AN INVOICE"] + ["-"] * (len(columns) - 2)
