
                          Table "public.users"
    Column     |          Type          | Collation | Nullable | Default 
---------------+------------------------+-----------+----------+---------
 id            | character varying(50)  |           | not null | 
 username      | character varying(100) |           | not null | 
 password_hash | character varying(255) |           | not null | 
 nickname      | character varying(100) |           |          | 
 avatar_url    | text                   |           |          | 
 created_at    | bigint                 |           | not null | 
 updated_at    | bigint                 |           | not null | 
 is_deleted    | boolean                |           |          | false
Indexes:
    "users_pkey" PRIMARY KEY, btree (id)
    "idx_users_username" btree (username) WHERE is_deleted = false
    "users_username_key" UNIQUE CONSTRAINT, btree (username)
Referenced by:
    TABLE "sessions" CONSTRAINT "fk_sessions_user" FOREIGN KEY (user_id) REFERENCES users(id)

                       Table "public.sessions"
   Column   |         Type          | Collation | Nullable | Default 
------------+-----------------------+-----------+----------+---------
 id         | character varying(50) |           | not null | 
 user_id    | character varying(50) |           | not null | 
 title      | text                  |           |          | 
 created_at | bigint                |           | not null | 
 updated_at | bigint                |           | not null | 
 is_deleted | boolean               |           | not null | false
Indexes:
    "sessions_pkey" PRIMARY KEY, btree (id)
    "idx_sessions_user_updated" btree (user_id, updated_at DESC) WHERE is_deleted = false
Foreign-key constraints:
    "fk_sessions_user" FOREIGN KEY (user_id) REFERENCES users(id)
Referenced by:
    TABLE "messages" CONSTRAINT "messages_session_id_fkey" FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE

                                Table "public.messages"
   Column   |         Type          | Collation | Nullable |          Default          
------------+-----------------------+-----------+----------+---------------------------
 id         | character varying(50) |           | not null | 
 session_id | character varying(50) |           | not null | 
 role       | character varying(10) |           | not null | 
 content    | text                  |           | not null | 
 feedback   | character varying(10) |           | not null | 'none'::character varying
 remark     | text                  |           |          | 
 created_at | bigint                |           | not null | 
 updated_at | bigint                |           | not null | 
Indexes:
    "messages_pkey" PRIMARY KEY, btree (id)
    "idx_messages_session_created" btree (session_id, created_at)
Check constraints:
    "messages_feedback_check" CHECK (feedback::text = ANY (ARRAY['none'::character varying, 'like'::character varying, 'dislike'::character varying]::text[]))
    "messages_role_check" CHECK (role::text = ANY (ARRAY['user'::character varying, 'assistant'::character varying]::text[]))
Foreign-key constraints:
    "messages_session_id_fkey" FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE