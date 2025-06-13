row = [x.strip().strip('"') for x in line.split(",")]
                    if len(row) >= len(columns) - 1:
                        row = row[:len(columns) - 1]
                        results.append([file.name] + row)
                        matched = True
                        break

                if not matched:
                    st.warning(f"Likely not invoice or could not parse {file.name}.")
                    st.text_area(f"Raw Output ({file.name})", csv_line)
                    results.append([file.name] + ["NOT AN INVOICE"] + ["-"] * (len(columns) - 2))

            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {e}")
                st.text_area(f"Raw Output ({file.name})", traceback.format_exc())
                results.append([file.name] + ["NOT AN INVOICE"] + ["-"] * (len(columns) - 2))

# ---------- DISPLAY RESULTS ----------
if results:
    df = pd.DataFrame(results, columns=columns)
    st.success("✅ All invoices processed!")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode()
    st.download_button("📥 Download Extracted Data", csv, "invoice_data.csv", "text/csv")
    st.balloons()
else:
    st.info("Upload one or more scanned invoices to get started.")
