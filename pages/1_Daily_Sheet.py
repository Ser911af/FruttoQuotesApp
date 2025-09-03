# ---------- Modo edición (todas las variables excepto la fecha) ----------
st.divider()
edit_mode = st.toggle("✏️ Modo edición (todo excepto fecha)", value=False,
                      help="Edita Shipper, Where, Product, OG/CV, Price, Volume Qty/Unit. La fecha permanece bloqueada.")

if edit_mode:
    # TODAS las columnas reales excepto fecha
    editable_cols = ["VendorClean", "Location", "Product", "organic", "Price", "volume_num", "volume_unit"]

    edit_df = day_df[["id", "cotization_date"] + editable_cols].copy()
    # Renombrar para que el editor sea legible (y luego revertimos antes de guardar)
    edit_df = edit_df.rename(columns={
        "VendorClean": "Shipper",
        "Location": "Where",
        "Price": "price"  # editor trabaja con numérico simple
    })

    col_config = {
        "id": st.column_config.TextColumn("ID", disabled=True),
        "cotization_date": st.column_config.DatetimeColumn("Date", format="MM/DD/YYYY", disabled=True),
        "Shipper": st.column_config.TextColumn("Shipper"),
        "Where": st.column_config.TextColumn("Where"),
        "Product": st.column_config.TextColumn("Product"),
        "organic": st.column_config.NumberColumn("OG/CV (1=OG,0=CV)", min_value=0, max_value=1, step=1),
        "price": st.column_config.NumberColumn("Price", min_value=0.0, step=0.01),
        "volume_num": st.column_config.NumberColumn("Volume Qty", min_value=0.0, step=0.01),
        "volume_unit": st.column_config.TextColumn("Volume Unit"),
    }

    st.caption("Edita los campos y presiona **Guardar cambios**.")
    edited_df = st.data_editor(
        edit_df,
        key="editor_all",
        num_rows="fixed",
        use_container_width=True,
        column_config=col_config,
        column_order=["id","cotization_date","Shipper","Where","Product","organic","price","volume_num","volume_unit"]
    )

    if st.button("💾 Guardar cambios", type="primary", use_container_width=True):
        orig = edit_df.set_index("id")[["Shipper","Where","Product","organic","price","volume_num","volume_unit"]]
        new  = edited_df.set_index("id")[["Shipper","Where","Product","organic","price","volume_num","volume_unit"]]

        changed_mask = (orig != new) & ~(orig.isna() & new.isna())
        dirty_ids = new.index[changed_mask.any(axis=1)].tolist()

        if not dirty_ids:
            st.success("No hay cambios por guardar.")
        else:
            payload = []
            for _id in dirty_ids:
                row = new.loc[_id].to_dict()

                # Conversión mínima segura
                for k in ["price", "volume_num"]:
                    try:
                        if row.get(k) not in (None, ""):
                            row[k] = float(row[k])
                    except Exception:
                        pass
                try:
                    if row.get("organic") not in (None, ""):
                        row["organic"] = int(row["organic"])
                except Exception:
                    pass

                # Revertir nombres a columnas reales de la tabla
                row["VendorClean"] = row.pop("Shipper", None)
                row["Location"]    = row.pop("Where", None)
                row["Price"]       = row.pop("price", None)

                # Quitar Nones para no sobreescribir con nulls si tu RLS lo restringe
                clean = {k: v for k, v in row.items() if v is not None}
                payload.append({"id": _id, **clean})

            try:
                from supabase import create_client
                SUPABASE_URL = st.secrets["SUPABASE_URL"]
                SUPABASE_KEY = st.secrets["SUPABASE_ANON_KEY"]
                sb = create_client(SUPABASE_URL, SUPABASE_KEY)

                for item in payload:
                    _id = item.pop("id")
                    sb.table("quotations").update(item).eq("id", _id).execute()

                st.success(f"Se actualizaron {len(payload)} registro(s). 🎉")
                st.balloons()

                # Refrescar en memoria las columnas derivadas más críticas
                id_map_price = {int(_id): p.get("Price") for _id, p in
                                ((int(x["id"]), x) for x in [{"id": k, **v} for k, v in new.to_dict(orient="index").items()])}
                mask = day_df["id"].isin(dirty_ids)
                # Actualiza Price y Price$
                day_df.loc[mask, "Price"] = day_df.loc[mask, "id"].map(id_map_price).fillna(day_df.loc[mask,"Price"])
                day_df["Price$"] = day_df["Price"].apply(_format_price)

            except Exception as e:
                st.error(f"Error al guardar cambios: {e}")
