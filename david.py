import shelve

if __name__ == "__main__":
    # a_c_index.shelve
    # d_f_index.shelve
    # g_i_index.shelve
    # j_l_index.shelve
    # m_o_index.shelve
    # p_r_index.shelve
    # s_u_index.shelve
    # v_z_index.shelve
    # misc_index.shelve
    count = 0
    with shelve.open('misc_index.shelve') as db:
        for key in db:
            count += 1
            print(key, db[key])
        print(count)
    db.close()
