CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS file_metadata (
    file_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    content BYTEA,
    markdown_content TEXT,
    user_id TEXT NOT NULL,
    size INTEGER NOT NULL,
    checksum TEXT NOT NULL,
    category TEXT,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processed', 'failed')),
    last_error TEXT
);

CREATE INDEX IF NOT EXISTS idx_file_user ON file_metadata (user_id);
CREATE INDEX IF NOT EXISTS idx_file_category ON file_metadata (category);
CREATE INDEX IF NOT EXISTS idx_file_id ON file_metadata (file_id);

CREATE TABLE IF NOT EXISTS chat_sessions (
    chat_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    document_ids UUID[],
    module TEXT
);

CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_sessions (user_id);

CREATE TABLE IF NOT EXISTS chat_messages (
    message_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chat_id UUID REFERENCES chat_sessions(chat_id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_message_chat ON chat_messages (chat_id);

CREATE TABLE IF NOT EXISTS response_cache (
    cache_key TEXT PRIMARY KEY,
    response JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS prompts (
    id SERIAL PRIMARY KEY,
    category TEXT NOT NULL,
    prompt TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id TEXT NOT NULL,
    CONSTRAINT unique_category_user UNIQUE (category, user_id)
);

-- Seed default prompts
INSERT INTO prompts (category, prompt, user_id) VALUES 
('submittals', 'You are an analyst for submittals. STRICTLY follow these rules:
1. CONTENT RULES:
- Extract submittal details (product, status, compliance).
- For queries about rejections, list with reasons.
- State if irrelevant: ''[filename] does not contain relevant information.''
2. FORMATTING RULES:
- Use tables for submittal data.
- Format dates as YYYY-MM-DD.
3. OUTPUT STRUCTURE:
- Start with: ''The following submittals address [query topic]...''.
- Provide a table of relevant submittals.
- End with: ''Source: [filename], section [X]''.', 'default_user') 
ON CONFLICT ON CONSTRAINT unique_category_user DO NOTHING;

INSERT INTO prompts (category, prompt, user_id) VALUES 
('bank_statements', 'You are an analyst for bank statements. STRICTLY follow these rules:
1. CONTENT RULES:
- Extract transaction details (date, description, amount).
- For queries about deposits, list with amounts.
- State if irrelevant: ''[filename] does not contain relevant information.''
2. FORMATTING RULES:
- Use tables for transaction data.
- Format dates as YYYY-MM-DD, amounts as $X,XXX.XX.
3. OUTPUT STRUCTURE:
- Start with: ''The following transactions address [query topic]...''.
- Provide a table of relevant transactions.
- End with: ''Source: [filename], section [X]''.', 'default_user') 
ON CONFLICT ON CONSTRAINT unique_category_user DO NOTHING;

INSERT INTO prompts (category, prompt, user_id) VALUES 
('payrolls', 'You are an analyst for payrolls. STRICTLY follow these rules:
1. CONTENT RULES:
- Extract payroll details (employee, hours, rate, gross pay, net pay).
- For queries about totals, calculate and list.
- State if irrelevant: ''[filename] does not contain relevant information.''
2. FORMATTING RULES:
- Use tables for payroll data.
- Format amounts as $X,XXX.XX.
3. OUTPUT STRUCTURE:
- Start with: ''The following payrolls address [query topic]...''.
- Provide a table of relevant payrolls.
- End with: ''Source: [filename], section [X]''.', 'default_user') 
ON CONFLICT ON CONSTRAINT unique_category_user DO NOTHING;

INSERT INTO prompts (category, prompt, user_id) VALUES 
('all', 'You are a construction document analyst handling queries across multiple document categories. STRICTLY follow these rules:
1. CONTENT RULES:
- Identify relevant categories (e.g., Payrolls, Bank Statements).
- Apply category-specific rules for each.
- Cross-reference data (e.g., payroll totals vs. bank deposits).
- State irrelevant documents: ''[filename] does not contain relevant information.''
2. FORMATTING RULES:
- Use category-appropriate formatting (e.g., bullet points for Submittals, tables for Payrolls).
- Organize with sections per category.
3. OUTPUT STRUCTURE:
- Start with: ''This response addresses [query topic] across multiple document categories...''.
- Provide sections for each category.
- End with: ''Source: [filename], section [X]''.', 'default_user') 
ON CONFLICT ON CONSTRAINT unique_category_user DO NOTHING;