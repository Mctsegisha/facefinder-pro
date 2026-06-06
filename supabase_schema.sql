-- Enable the pgvector extension to work with embedding vectors
create extension if not exists vector;

-- Create the photos table to store image URLs and their associated face embeddings
create table if not exists photos (
    id uuid default gen_random_uuid() primary key,
    url text not null,
    category text not null default 'Candid Social Interaction',
    face_box jsonb, -- stores {x, y, w, h} bounding box coordinates
    timestamp text,
    embedding vector(128), -- 128-dimensional embedding from face-api.js
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create an HNSW index for fast similarity search using cosine distance
create index if not exists photos_embedding_hnsw_idx 
on photos using hnsw (embedding vector_cosine_ops);

-- Create a helper function (RPC) to perform cosine similarity searches
create or replace function match_photos (
  query_embedding vector(128),
  match_threshold float,
  match_count int,
  filter_category text default 'All'
)
returns table (
  id uuid,
  url text,
  category text,
  face_box jsonb,
  timestamp text,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    photos.id,
    photos.url,
    photos.category,
    photos.face_box,
    photos.timestamp,
    1 - (photos.embedding <=> query_embedding) as similarity
  from photos
  where 
    -- Filter out weak similarities based on threshold
    1 - (photos.embedding <=> query_embedding) > match_threshold
    -- Apply category filter if provided
    and (filter_category = 'All' or photos.category = filter_category)
  order by 
    photos.embedding <=> query_embedding
  limit match_count;
end;
$$;
