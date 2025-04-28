SELECT * FROM Artist;
SELECT * FROM Album WHERE ArtistId = (SELECT ArtistId FROM Artist WHERE Name = 'AC/DC');
SELECT * FROM Track WHERE GenreId = (SELECT GenreId FROM Genre WHERE Name = 'Rock');
SELECT SUM(Milliseconds) FROM Track;
SELECT * FROM Customer WHERE Country = 'Canada';
SELECT COUNT(*) FROM Track WHERE AlbumId = 5;
SELECT COUNT(*) FROM Invoice;
SELECT * FROM Track WHERE Milliseconds > 300000;
SELECT CustomerId, SUM(Total) AS TotalPurchase FROM Invoice GROUP BY CustomerId ORDER BY TotalPurchase DESC LIMIT 5;
SELECT COUNT(*) FROM Employee
SELECT * FROM Artist WHERE Name LIKE 'A%';
SELECT * FROM Album WHERE Title LIKE '%Live%';
SELECT * FROM Track WHERE Composer IS NULL;
SELECT * FROM Customer WHERE Country = 'USA';
SELECT * FROM Customer WHERE Country = 'Brazil';
SELECT * FROM Employee WHERE Title = 'Sales Support Agent';
SELECT * FROM Invoice WHERE Total > 10;
SELECT * FROM Track WHERE Milliseconds > 500000;
SELECT * FROM InvoiceLine WHERE Quantity > 1;
SELECT * FROM Customer WHERE Company IS NOT NULL;
SELECT 
    e.EmployeeId,
    e.FirstName || ' ' || e.LastName AS EmployeeName,
    SUM(i.Total) AS TotalSales,
    COUNT(DISTINCT c.CustomerId) AS CustomerCount
FROM Employee e
JOIN Customer c ON e.EmployeeId = c.SupportRepId
JOIN Invoice i ON c.CustomerId = i.CustomerId
WHERE i.InvoiceDate >= DATE('now', '-3 month')
GROUP BY e.EmployeeId, e.FirstName, e.LastName
ORDER BY TotalSales DESC
LIMIT 5;
