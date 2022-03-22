import React, {useEffect, useState} from "react";
import {
    Card,
    SimpleGrid,
    Text,
    Button,
    useMantineTheme,
    Container
} from "@mantine/core";
import axios from "axios";

export function Articles() {
    const [articles, setArticles] = useState([]);
    const theme = useMantineTheme();
    const secondaryColor = theme.colorScheme === "dark" ? theme.colors.dark[1] : theme.colors.gray[7];

    useEffect(() => {
        axios.get("http://localhost:8000/articles")
            .then(res => {
                setArticles(res.data);
            });
    }, []);

    const cards = articles.map((article) => (
        <Card shadow="sm" key={article.title} style={{minWidth: 240}}>
            <div
                style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    marginBottom: 10
                }}
            >
                <Text weight={500}>{article.title}</Text>
            </div>
            <Text size="sm" style={{color: secondaryColor, minHeight: 140}}>
                {article.content}
            </Text>
            <Button
                size="sm"
                variant="light"
                color="cyan"
                fullWidth
                style={{marginTop: 10}}
            >
                Read more
            </Button>
        </Card>
    ));

    return (
        <div style={{backgroundColor: theme.colors.gray[0]}}>
            <Container style={{paddingTop: 50, paddingBottom: 50}} size="md">
                <SimpleGrid grow>
                    {cards}
                </SimpleGrid>
            </Container>
        </div>
    );
}
